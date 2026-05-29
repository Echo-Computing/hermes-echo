"""Generation agent — searches literature and generates hypotheses from findings."""

import asyncio
from loguru import logger
from hermes_cli.agents.echo.research.state import ResearchState
from hermes_cli.agents.echo.research.models import Hypothesis
from hermes_cli.tools.ollama_client import OllamaClient
from hermes_cli.agents.echo.tools.web_tools import search_web, fetch_url


GENERATION_SYSTEM_PROMPT = """You are the Generation Agent in a multi-agent scientific research system (inspired by Google's Co-Scientist).

Your job is to:
1. Review the provided literature/search results
2. Synthesize findings into specific, testable hypotheses
3. For each hypothesis, explain the mechanism (HOW it would work)
4. Cite supporting evidence

Generate {num_hypotheses} hypotheses. Each should be:
- Specific and falsifiable
- Novel (not just restating known facts)
- Mechanistically explained (HOW, not just WHAT)

Return JSON:
{{
  "hypotheses": [
    {{
      "title": "One-line hypothesis summary",
      "description": "Full hypothesis explanation (2-4 sentences)",
      "mechanism": "The underlying mechanism — how and why this would work",
      "evidence": ["Source URL or finding that supports this", ...],
      "confidence": 0.0-1.0
    }}
  ]
}}"""


async def run_generation(state: ResearchState) -> ResearchState:
    """Generation node: search literature and generate hypotheses."""

    config = state.get("config", {})
    ollama_config = config.get("ollama", {})
    research_config = config.get("research", {})

    num_hypotheses = research_config.get("hypotheses_per_round", 5)
    round_num = state.get("current_round", 1)

    logger.info(
        "Generation: searching literature and generating hypotheses (round {})".format(round_num),
        extra={"category": "RESEARCH"},
    )

    client = OllamaClient(
        api_url=ollama_config.get("api_url", "http://localhost:11434/api/chat"),
        model=ollama_config.get("model", "kimi-k2.6:cloud"),
        timeout=ollama_config.get("timeout", 120),
        retry=ollama_config.get("retry", 3),
        temperature=0.7,
    )

    try:
        # --- Step 1: Search literature for each sub-question ---
        sub_questions = state.get("sub_questions", [state["research_goal"]])
        all_search_text = []

        for question in sub_questions[:3]:  # Limit to avoid too many searches
            try:
                results = await asyncio.to_thread(search_web, query=question, limit=5)
                if results:
                    all_search_text.append("## Search: {}".format(question))
                    all_search_text.append(results)
                    all_search_text.append("")
            except Exception as e:
                logger.warning(
                    "Search failed for '{}': {}".format(question[:50], e),
                    extra={"category": "RESEARCH"},
                )
                # Continue — don't fail the whole round for one search error

        # Accumulate across rounds
        if "search_results" not in state:
            state["search_results"] = []
        state["search_results"].extend([
            {"round": round_num, "query": q, "results_preview": True}
            for q in sub_questions[:3]
        ])

        search_context = "\n".join(all_search_text) if all_search_text else state["research_goal"]

        # --- Step 2: Generate hypotheses from literature ---
        # Add context from prior round's surviving hypotheses
        prior_context = ""
        existing_hypotheses = state.get("hypotheses", [])
        alive = [h for h in existing_hypotheses if h.get("status") == "alive"]
        if alive and round_num > 1:
            prior_context = "\n\nSurviving hypotheses from previous round (build on or diverge from these):\n"
            prior_context += "\n".join(
                "- [{elo:.0f}] {title}: {desc}".format(
                    elo=h.get("elo_rating", 1500),
                    title=h.get("title", ""),
                    desc=h.get("description", "")[:150],
                )
                for h in alive[:5]
            )

        prompt = """Research Goal: {goal}

Sub-questions to investigate:
{sub_questions}

Literature Search Results:
{search_results}
{prior_context}

Generate {num_hypotheses} specific, testable hypotheses based on the above.""".format(
            goal=state["research_goal"],
            sub_questions="\n".join("- {}".format(q) for q in sub_questions),
            search_results=search_context[:6000],  # Truncate for context window
            prior_context=prior_context,
            num_hypotheses=num_hypotheses,
        )

        system_prompt = GENERATION_SYSTEM_PROMPT.format(num_hypotheses=num_hypotheses)

        response = await client.chat(prompt, system_prompt, temperature=0.7)

        # --- Step 3: Parse hypotheses ---
        import json
        import re

        # Extract JSON from response — try multiple strategies
        hypotheses = []
        data = None
        clean_response = response

        # Strategy 0: strip markdown code blocks first
        md_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if md_match:
            clean_response = md_match.group(1).strip()

        # Strategy 1: find JSON object with hypotheses key
        json_match = re.search(r'{"hypotheses"\s*:\s*\[.*?\]\s*\}', clean_response, re.DOTALL)
        if not json_match:
            # Strategy 2: find any JSON object
            json_match = re.search(r"\{.*\}", clean_response, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                # Strategy 3: try to fix truncated JSON by closing braces
                truncated = json_match.group()
                # Count open vs close braces and balance
                open_count = truncated.count("{") - truncated.count("}")
                if open_count > 0:
                    truncated += "}" * open_count
                try:
                    data = json.loads(truncated)
                except json.JSONDecodeError:
                    data = None

        if data:
            raw_hypotheses = data.get("hypotheses", [])
            for h in raw_hypotheses:
                try:
                    hypotheses.append(Hypothesis(
                        title=str(h.get("title", "")),
                        description=str(h.get("description", "")),
                        mechanism=str(h.get("mechanism", "")),
                        evidence=h.get("evidence", []) if isinstance(h.get("evidence"), list) else [],
                        elo_rating=1500.0,
                        round_created=round_num,
                        status="alive",
                    ))
                except Exception as inner_e:
                    logger.warning("Failed to parse individual hypothesis: {}".format(inner_e), extra={"category": "RESEARCH"})

        # --- Strategy 4: Parse structured markdown / numbered text ---
        if not hypotheses and response:
            split_patterns = [
                r'(?:^|\n)\s*\d+\.\s+',
                r'(?:^|\n)\s*[-*]\s+',
                r'(?:^|\n)\s*#{1,3}\s+',
                r'(?:^|\n)\s*\*\*[^*]+\*\*\s*[:\-]?\s*',
                r'(?:^|\n)\s*Hypothesis\s+\d+[:\.]?\s*',
            ]
            parsed_hyps = []
            for pattern in split_patterns:
                parts = re.split(pattern, response)
                if len(parts) >= 3:
                    for part in parts[1:]:
                        if len(part.strip()) < 30:
                            continue
                        title = part.strip().split('\n')[0][:100]
                        mech_match = re.search(r'(?:[Mm]echanism|[Hh]ow it works)[:\-]?\s*(.+?)(?=\n\n|\n\d+\.|$)', part, re.DOTALL)
                        mechanism = mech_match.group(1).strip()[:300] if mech_match else ""
                        evidence = re.findall(r'(?:[Ee]vidence|[Ss]ource|[Cc]itation)[:\-]?\s*(.+?)(?=\n|\n\n|$)', part)
                        parsed_hyps.append({
                            "title": title,
                            "description": part[:400].strip(),
                            "mechanism": mechanism,
                            "evidence": evidence[:3] if evidence else [],
                        })
                    if len(parsed_hyps) >= 2:
                        break
            if parsed_hyps:
                for h in parsed_hyps[:num_hypotheses]:
                    hypotheses.append(Hypothesis(
                        title=h["title"],
                        description=h["description"],
                        mechanism=h["mechanism"] or "Mechanism inferred from structured text",
                        evidence=h["evidence"],
                        elo_rating=1500.0,
                        round_created=round_num,
                        status="alive",
                    ))
                logger.info(
                    "Generation: parsed {} hypotheses from structured text (strategy 4)".format(len(hypotheses)),
                    extra={"category": "RESEARCH"},
                )

        # --- Strategy 5: Regex field extraction from plain text ---
        if not hypotheses and response:
            title_matches = re.findall(r"(?:[Tt]itle|[Hh]ypothesis)[:\-]?\s*[\"\']?([^\"\n]{10,120})[\"\']?", response)
            desc_matches = re.findall(r"(?:[Dd]escription|[Ss]ummary|[Ww]hat)[:\-]?\s*[\"\']?([^\"\n]{20,300})[\"\']?", response)
            mech_matches = re.findall(r"(?:[Mm]echanism|[Hh]ow)[:\-]?\s*[\"\']?([^\"\n]{20,300})[\"\']?", response)
            if title_matches:
                for i, title in enumerate(title_matches[:num_hypotheses]):
                    desc = desc_matches[i] if i < len(desc_matches) else ""
                    mech = mech_matches[i] if i < len(mech_matches) else ""
                    hypotheses.append(Hypothesis(
                        title=title.strip(),
                        description=desc.strip() if desc else title.strip(),
                        mechanism=mech.strip() if mech else "Mechanism not explicitly stated",
                        evidence=[],
                        elo_rating=1500.0,
                        round_created=round_num,
                        status="alive",
                    ))
                if hypotheses:
                    logger.info(
                        "Generation: extracted {} hypotheses via regex fields (strategy 5)".format(len(hypotheses)),
                        extra={"category": "RESEARCH"},
                    )

        # --- Final Fallback: split response into paragraph-based hypotheses ---
        if not hypotheses and response:
            # Split by double newlines (paragraphs) if response is substantial
            paragraphs = [p.strip() for p in response.split('\n\n') if len(p.strip()) >= 40]
            if len(paragraphs) >= 2:
                for i, para in enumerate(paragraphs[:num_hypotheses]):
                    title = para.split('\n')[0][:100] if '\n' in para else para[:100]
                    hypotheses.append(Hypothesis(
                        title=title,
                        description=para[:400],
                        mechanism="Paragraph-derived hypothesis from unstructured response",
                        evidence=[],
                        elo_rating=1500.0,
                        round_created=round_num,
                        status="alive",
                    ))
                logger.info(
                    "Generation: split response into {} paragraph hypotheses (final fallback)".format(len(hypotheses)),
                    extra={"category": "RESEARCH"},
                )
            else:
                desc = response[:500] if response else "No content generated"
                hypotheses.append(Hypothesis(
                    title="Synthesized hypothesis (round {})".format(round_num),
                    description=desc,
                    mechanism="Generated from literature synthesis",
                    evidence=[],
                    elo_rating=1500.0,
                    round_created=round_num,
                    status="alive",
                ))

        # Merge with existing hypotheses
        all_hypotheses = state.get("hypotheses", [])
        all_hypotheses.extend([h.to_dict() for h in hypotheses])
        state["hypotheses"] = all_hypotheses

        logger.info(
            "Generation: created {} hypotheses (total: {})".format(
                len(hypotheses), len(all_hypotheses)
            ),
            extra={"category": "RESEARCH"},
        )

    except Exception as e:
        logger.error("Generation error: {}".format(e), extra={"category": "RESEARCH"})
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append({"node": "generation", "error": str(e)})

    finally:
        await client.close()

    return state
