# Third-party notices

TeNet contains or adapts components from the projects below. The repository's
MIT license applies to the original TeNet contributions; third-party material
remains subject to its respective terms and notices.

## Prompt-DT

- Source: <https://github.com/mxu34/prompt-dt>
- Paper: *Prompting Decision Transformer for Few-Shot Policy Generalization*
  (Xu et al., ICML 2022)

Prompt-DT provides the research-code backbone for trajectory processing,
prompt construction, training, and evaluation. TeNet extends that backbone
with language-conditioned policy synthesis and grounding objectives. Copyright
in Prompt-DT material remains with its respective authors; consult the upstream
repository for its terms.

## Decision Transformer

- Source: <https://github.com/kzl/decision-transformer>
- License: MIT
- Included license: [`licenses/DECISION_TRANSFORMER_LICENSE`](licenses/DECISION_TRANSFORMER_LICENSE)

Prompt-DT and the trajectory transformer build on Decision Transformer.

## Meta-World

- Source: <https://github.com/Farama-Foundation/Metaworld>
- License: MIT
- Included license: [`licenses/METAWORLD_LICENSE`](licenses/METAWORLD_LICENSE)

The bundled `metaworld/` directory contains the benchmark environments,
assets, and scripted expert policies used by this release.

## Hugging Face Transformers

- Source: <https://github.com/huggingface/transformers>
- License: Apache License 2.0
- Included license: [`licenses/TRANSFORMERS_LICENSE`](licenses/TRANSFORMERS_LICENSE)

`tenet/trajectory_gpt2.py` is derived from Hugging Face's GPT-2 modeling code
and retains its upstream copyright and license header.

## Llama 3

TeNet downloads `meta-llama/Meta-Llama-3-8B` at runtime. Model weights are not
distributed in this repository. Users are responsible for obtaining access and
complying with the model terms presented by its provider.
