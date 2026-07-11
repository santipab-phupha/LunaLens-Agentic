VIDEO_FACTS_PROMPT = """
You are a careful video understanding model. You will receive evenly sampled frames from one short video.

Describe only visible content from the frames. Do not invent people, objects, text, brands, locations, relationships, causes, or events that are not visually supported.

Return only a JSON object with this exact schema:
{
  "main_subjects": ["visible subject or object"],
  "visible_actions": ["visible action"],
  "setting": "visible setting or unknown",
  "mood": "brief visible mood or neutral",
  "important_visual_details": ["specific visible detail"],
  "uncertain_details": ["anything unclear or not safely inferable"],
  "caption_seed": "one factual sentence suitable as a caption base"
}

Keep the facts concise and grounded in visual evidence.
"""

STYLE_CAPTION_PROMPT = """
Generate candidate video captions from structured visual facts.

Rules:
- Use only the supplied facts.
- Do not add people, objects, text, brands, places, emotions, or events not in the facts.
- All captions must be in English.
- Each caption must be one sentence.
- Each caption should be around 8 to 22 words.
- Do not include markdown, labels, quotation marks, or style names inside captions.
- Generate exactly the requested number of candidates for each requested style.

Style definitions:
- formal: professional, objective, factual.
- sarcastic: dry, ironic, lightly mocking, but still faithful.
- humorous_tech: funny with technology or programming references.
- humorous_non_tech: funny everyday humor with no technical jargon.

Return only a JSON object mapping each requested style to a list of candidate captions.
"""

SELF_JUDGE_PROMPT = """
Select final video captions for a competition judge.

Evaluate each candidate for:
1. Faithfulness to the video facts.
2. Match to the requested style.
3. Concision and one-sentence format.

Choose the best candidate for each style, or rewrite it if needed. Faithfulness is more important than humor.

Rules:
- Use only the supplied video facts.
- All final captions must be in English.
- Include every requested style exactly once.
- Each final caption must be one sentence and around 8 to 22 words.
- Do not include markdown, labels, quotation marks, or style names inside caption text.

Return only this JSON object:
{
  "captions": {
    "style_name": "final caption"
  }
}
"""
