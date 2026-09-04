# Probe Scenarios

Curated contrastive prompt pairs used for activation patching in `mcp_eval.interp`.

Each YAML file holds one category. A `scenarios` list contains entries of the
form:

```yaml
scenarios:
  - id: weather_001
    query: "What is the weather in Tokyo tomorrow?"
    intended_tool: get_weather_forecast
    tools:
      - name: get_weather_forecast
        description: "Return hourly forecast for a city."
        parameters:
          type: object
          properties:
            city: {type: string}
          required: [city]
      - name: search_papers
        description: "Search academic papers by query."
        parameters:
          type: object
          properties:
            query: {type: string}
          required: [query]
```

Conventions
-----------

* `intended_tool` MUST match exactly one tool name in `tools`.
* Each scenario has between 2 and 4 tools (paper §3.1 setup).
* All tools follow the OpenAI function-calling schema so they can be reused via
  `mcp_eval.tools.sources.JSONToolSource`.
* Categories used by the paper: **weather**, **paper_search**, **stock**, **web_search**.
* Cross-category pairs (e.g. weather vs paper_search) form the contrastive
  pairs `(s_A, s_B)` used by `mcp_eval.interp.patching`.

To add a new probe scenario, simply append to the appropriate YAML file. Keep
identifiers stable so cached intermediate results remain valid.
