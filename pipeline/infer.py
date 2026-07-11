from pathlib import Path
from .iragent import IRAgent


input_path = Path("/root/autodl-tmp/AgenticIR/output_qx/haze+motion blur+low resolution/001-260131_211504/result.png").resolve()
output_dir = Path("output").resolve()

agent = IRAgent(
    input_path=input_path, output_dir=output_dir,
    evaluate_degradation_by="depictqa",
    with_retrieval=True,
    with_reflection=True,
    reflect_by="depictqa",
    with_rollback=True,
    silent=False
)

manual_plan = ['brightening']
agent.run(plan=manual_plan)
