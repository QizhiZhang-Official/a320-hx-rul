# gpu_monitor.py
import time
import sys
import pynvml

def safe(func, *args, default=None):
    try:
        return func(*args)
    except Exception:
        return default

def main():
    try:
        pynvml.nvmlInit()
    except Exception as e:
        print(f"NVML init failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 列宽已为 MEM 的 X.X/X.XGB 格式重新对齐
    fmt = "{:<8} {:<30} {:>5} {:>13} {:>5} {:>6} {:>4}"
    header = fmt.format("TIME", "GPU_NAME", "GPU%", "MEM", "TEMP", "POWER", "FAN")
    print(header)
    print("-" * len(header))

    try:
        while True:
            ts = time.strftime("%H:%M:%S")
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                name = name.decode() if isinstance(name, bytes) else name
                name = name[:29]

                util = safe(pynvml.nvmlDeviceGetUtilizationRates, h)
                gpu_pct = f"{util.gpu}%" if util is not None else "-"

                mem = safe(pynvml.nvmlDeviceGetMemoryInfo, h)
                if mem is not None:
                    used_gb = mem.used / (1024**3)
                    total_gb = mem.total / (1024**3)
                    mem_str = f"{used_gb:.1f}/{total_gb:.1f}GB"
                else:
                    mem_str = "-"

                temp = safe(pynvml.nvmlDeviceGetTemperature, h, pynvml.NVML_TEMPERATURE_GPU)
                temp_str = f"{temp}°C" if temp is not None else "-"

                power_mw = safe(pynvml.nvmlDeviceGetPowerUsage, h)
                power_w = f"{power_mw/1000:.1f}W" if power_mw is not None else "-"

                fan = safe(pynvml.nvmlDeviceGetFanSpeed, h)
                fan_str = f"{fan}%" if fan is not None else "-"

                print(fmt.format(ts, name, gpu_pct, mem_str, temp_str, power_w, fan_str))
            print("-" * len(header))
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()