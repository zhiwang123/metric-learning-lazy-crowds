import subprocess

programs_to_run = [
    {
        'program': 'multiprocess_runner.py',
        'params': ['--config', 'examples/configs/exp3/noise_0.yaml']
    },
    {
        'program': 'multiprocess_runner.py',
        'params': ['--config', 'examples/configs/exp3/noise_1.yaml']
    },
    {
        'program': 'multiprocess_runner.py',
        'params': ['--config', 'examples/configs/exp3/noise_2.yaml']
    },
    {
        'program': 'multiprocess_runner.py',
        'params': ['--config', 'examples/configs/exp3/noise_3.yaml']
    }
]

for program_info in programs_to_run:
    program = program_info['program']
    params = program_info['params']
    try:
        command = ['python', program] + params
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {program}: {e}")
    except FileNotFoundError as e:
        print(f"Error: {program} not found. Please check the filename or path.")

print("All programs have been executed.")
