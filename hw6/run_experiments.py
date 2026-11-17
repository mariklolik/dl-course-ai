import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from analyze import read_summary


def start_server(name: str) -> subprocess.Popen:
    log = Path('outputs') / f'{name}-serve.log'
    with log.open('a') as handle:
        process = subprocess.Popen(
            ['bash', f'configs/{name}.sh'], stdout=handle,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'{name}: сервер завершился, см. {log}')
        try:
            with urllib.request.urlopen(os.environ['URL'] + '/health', timeout=2):
                return process
        except (OSError, TimeoutError):
            time.sleep(3)
    stop_server(process)
    raise TimeoutError(f'{name}: сервер не запустился за 900 секунд')


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def measure(group: str, name: str, rpm: int) -> None:
    output = Path('outputs') / f'{group}-{name}-{rpm}'
    export = output / 'profile/profile_export_aiperf.json'
    if export.exists() and read_summary(export)['complete']:
        return
    script = ['bash', 'load_test.sh', str(rpm), str(output)] if group == 'load' else [
        'bash', 'benchmark.sh', str(output), str(rpm),
    ]
    with Path(f'{output}.log').open('w') as handle:
        completed = subprocess.run(script, stdout=handle, stderr=subprocess.STDOUT)
    result = {'command': script, 'exit_code': completed.returncode}
    if export.exists():
        result.update(read_summary(export))
    Path(f'{output}.receipt.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'run': str(output), **result}), flush=True)
    if completed.returncode:
        raise RuntimeError(f'Ошибка замера: {output}')


def main() -> None:
    os.environ['URL'] = f'http://127.0.0.1:{os.environ.get("PORT", "8616")}'
    os.environ['PATH'] = f'{Path(".benchmark/bin").resolve()}:{os.environ["PATH"]}'
    Path('outputs').mkdir(exist_ok=True)
    stage, *names = sys.argv[1:]
    rates = [120] if stage == 'fixed' else [int(rpm) for rpm in os.environ['RPM'].split(',')]
    for name in names:
        process = start_server(name)
        try:
            for rpm in rates:
                measure(stage, name, rpm)
        finally:
            stop_server(process)


if __name__ == '__main__':
    main()
