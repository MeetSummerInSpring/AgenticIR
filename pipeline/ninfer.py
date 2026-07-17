from pathlib import Path
from .iragent import IRAgent

# Define the input directory and output directory
input_dir = Path("/root/autodl-tmp/AgenticIR/dataset/QXLQ/d2/rain+haze").resolve()
output_dir = Path("output_qx/rain+haze").resolve()

# Loop through images 001.png to 010.png
for i in range(1, 11):
    input_file = input_dir / f"{i:03}.png"  # Format numbers as 001, 002, ..., 010
    if not input_file.exists():
        print(f"File {input_file} does not exist. Skipping.")
        continue

    print(f"Processing {input_file}...")

    agent = IRAgent(
        input_path=input_file, output_dir=output_dir,
        evaluate_degradation_by="depictqa",
        with_retrieval=True,
        with_reflection=True,
        reflect_by="depictqa",
        with_rollback=True,
        silent=False
    )

    # Run the agent for the current file
    # manual_plan = ['denoising', 'defocus deblurring', 'motion deblurring']
    # manual_plan = ['deraining', 'denoising', 'super-resolution']
    # manual_plan = ['dehazing', 'motion deblurring', 'super-resolution']
    manual_plan = ['deraining', 'dehazing']
    agent.run(plan=manual_plan)

print("Processing complete.")