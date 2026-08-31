Warm Context Vault measurements, gesicht (M3 Ultra 512GB), 2026-08-31.
Model: ~/glm53flash/builds/GLM-5.3-Flash-vlm-q4-quasar (320B-A18B q4)
Env:   MLX_VLM_GLM5_FUSED_KDA=1 MLX_VLM_GLM5_QPROJ=1
       MLX_MAX_MB_PER_BUFFER=2048 MLX_MAX_OPS_PER_BUFFER=100000
       vault arms add MLX_VLM_GLM5_VAULT=1 MLX_VLM_GLM5_VAULT_STRIDE=2048
Protocol: greedy, temperature 0, max_tokens 16, fresh process per file.
  store = first sighting of the document (cold, lays the ladder)
  warm  = same document, DIFFERENT question
  warm2 = same document, third question (steady state)

vaultoff_16k.json          baseline, all three arms cold
vault_16k_warmpages.json   vault on, warm page cache
vault_16k.json             vault on, FIRST process of the session (cold page
                           cache, load 41.0s) -- store arm is contaminated,
                           kept only as the page-cache evidence
vaultoff_32k.json          baseline
vault_32k.json             vault on
twobox_probe_32k.json      Stage 3 serialization probe (no peer, no model)
