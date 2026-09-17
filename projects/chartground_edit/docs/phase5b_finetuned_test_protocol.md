# Phase 5B fine-tuned test protocol

- Selected checkpoint: `step960`
- Projection checkpoint SHA-256: `64c0d109d2985893ba1f2ba4c4fe7acc4265dc758e5d56ecb6ac8e2aa791f41e`
- Prompt: P2 `target_only_zh`
- Prompt registry SHA-256: `dc822a33b84b1cdfb72f84bd5288f0ebb37626980496107c5e30d4c4c26217c0`
- P2 template SHA-256: `37a785d086a80fef21fd69014670b3892acad5c379722cb79658fb837a923806`
- synthetic_v1 manifest SHA-256: `ebad55fd98356204e572ffe6607a16a34c9dde8a916a9977a7f08bc4aed2ba82`
- Test sample-ID list SHA-256: `3061387f0012b13c2fd998d81819df0e7648e8e47ae9ad86ff06490ecfe2e5c4`
- Base model: `OpenGVLab/InternVL3-2B`
- Base revision: `899155015275a9b7338c7f4677e19c784e0e5a21`
- Sa2VA revision: `15837dcaecc304714a1f0f069e74f47e47521c7f`
- Full PTH SHA-256: `5aa030f3203487281abcb57d7dbed72bba618b9f08859e1c020085454b2822e6`
- Expected test samples: 64
- Expected backend calls: 64

This is the only fine-tuned test run. The checkpoint must not be reselected, no
training may be restarted, and test results must not be used to modify the
model, Prompt, data, or checkpoint.
