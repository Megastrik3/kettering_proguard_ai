from ultralytics.utils.benchmarks import benchmark
import model_metrics



if __name__ == "__main__":
    selected_model = model_metrics.getModels()
    # Benchmark on GPU
    benchmark(model=selected_model, data="./datasets/bus-aps/data.yaml", imgsz=640, device=0, verbose=True)
