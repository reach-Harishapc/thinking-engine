#!/usr/bin/env python3
"""
Thinking Engine — Multi-Platform Testing Framework
Author: Harish
Purpose:
    Comprehensive testing across CPU, GPU (NVIDIA), MPS (Apple Silicon), and Quantum computing platforms.
    Supports benchmarking, performance comparison, and backend validation.
"""

import os
import sys
import time
import platform
import psutil
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import backend types from the core backend module
from core.backend import BackendType, ComputeBackend, BackendManager

@dataclass
class BenchmarkResult:
    backend: str
    operation: str
    duration_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None
    additional_metrics: Optional[Dict[str, Any]] = None

@dataclass
class PlatformInfo:
    os: str
    architecture: str
    python_version: str
    available_backends: List[str]
    hardware_info: Dict[str, Any]

class MultiPlatformTester:
    """Comprehensive testing framework for multiple computing backends."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.platform_info = self._detect_platform()

    def _detect_platform(self) -> PlatformInfo:
        """Detect available computing platforms and hardware."""
        available_backends = [BackendType.CPU.value]

        # Check for CUDA (NVIDIA GPU)
        try:
            import torch
            if torch.cuda.is_available():
                available_backends.append(BackendType.CUDA.value)
        except ImportError:
            pass

        # Check for MPS (Apple Silicon)
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                available_backends.append(BackendType.MPS.value)
        except ImportError:
            pass

        # Check for Metal (macOS GPU)
        try:
            import torch
            if torch.backends.mps.is_available():
                available_backends.append(BackendType.METAL.value)
        except ImportError:
            pass

        # Check for IBM Quantum
        try:
            import qiskit
            available_backends.append(BackendType.QUANTUM_IBM.value)
        except ImportError:
            pass

        # Hardware info
        hardware_info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        }

        # GPU info if available
        try:
            import torch
            if torch.cuda.is_available():
                hardware_info["cuda_devices"] = torch.cuda.device_count()
                hardware_info["cuda_device_names"] = [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ]
        except ImportError:
            pass

        return PlatformInfo(
            os=platform.system(),
            architecture=platform.machine(),
            python_version=platform.python_version(),
            available_backends=available_backends,
            hardware_info=hardware_info
        )

    def test_sparse_encoding(self, backend: BackendType, test_data: List[str]) -> BenchmarkResult:
        """Test sparse encoding performance across backends."""
        from core.utils import encode_text_to_sparse_representation

        start_time = time.time()
        memory_start = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        cpu_start = psutil.cpu_percent(interval=None)

        try:
            results = []
            for text in test_data:
                vector = encode_text_to_sparse_representation(text)
                results.append(vector)

            duration = time.time() - start_time
            memory_end = psutil.Process().memory_info().rss / (1024 * 1024)
            cpu_end = psutil.cpu_percent(interval=None)

            return BenchmarkResult(
                backend=backend.value,
                operation="sparse_encoding",
                duration_seconds=duration,
                memory_usage_mb=memory_end - memory_start,
                cpu_usage_percent=cpu_end,
                success=True,
                additional_metrics={
                    "samples_processed": len(test_data),
                    "avg_vector_size": sum(len(v) for v in results) / len(results),
                    "total_characters": sum(len(t) for t in test_data)
                }
            )

        except Exception as e:
            return BenchmarkResult(
                backend=backend.value,
                operation="sparse_encoding",
                duration_seconds=time.time() - start_time,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                success=False,
                error_message=str(e)
            )

    def test_model_training(self, backend: BackendType, dataset_path: str) -> BenchmarkResult:
        """Test model training performance with real backend implementations."""
        from core.learning_manager import MultiPlatformLearningManager
        from core.utils import create_training_dataset_from_directory

        start_time = time.time()
        memory_start = psutil.Process().memory_info().rss / (1024 * 1024)
        cpu_start = psutil.cpu_percent(interval=None)

        try:
            # Load training data
            training_data = create_training_dataset_from_directory(dataset_path)

            # Encode data
            from core.utils import encode_text_to_sparse_representation
            import numpy as np

            encoded_samples = []
            for sample in training_data:
                vector = encode_text_to_sparse_representation(sample)
                encoded_samples.append(vector)

            encoded_array = np.array(encoded_samples)

            # Initialize backend-specific learning manager
            learning_manager = MultiPlatformLearningManager(backend)

            # Train model using real backend implementations
            num_epochs = 1000  # Increased epochs as requested00
            learning_result = learning_manager.learn(encoded_array, num_epochs)

            duration = time.time() - start_time
            memory_end = psutil.Process().memory_info().rss / (1024 * 1024)
            cpu_end = psutil.cpu_percent(interval=None)

            return BenchmarkResult(
                backend=backend.value,
                operation="model_training",
                duration_seconds=duration,
                memory_usage_mb=memory_end - memory_start,
                cpu_usage_percent=cpu_end,
                success=True,
                additional_metrics={
                    "samples_trained": len(training_data),
                    "encoded_dimensions": encoded_array.shape,
                    "total_characters": sum(len(s) for s in training_data),
                    "training_progress": learning_result["training_progress"],
                    "final_loss": learning_result["final_loss"],
                    "final_accuracy": learning_result["final_accuracy"],
                    "backend_used": backend.value,
                    "learning_config": learning_result["training_progress"]["learning_config"]
                }
            )

        except Exception as e:
            return BenchmarkResult(
                backend=backend.value,
                operation="model_training",
                duration_seconds=time.time() - start_time,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                success=False,
                error_message=str(e)
            )

    def test_quantum_simulation(self, backend: BackendType) -> BenchmarkResult:
        """Test quantum computing simulation capabilities."""
        start_time = time.time()
        memory_start = psutil.Process().memory_info().rss / (1024 * 1024)
        cpu_start = psutil.cpu_percent(interval=None)

        try:
            from qiskit import QuantumCircuit, transpile
            from qiskit.providers.basic_provider import BasicSimulator
            from qiskit.visualization import circuit_drawer

            # Create a simple quantum circuit
            qc = QuantumCircuit(2, 2)
            qc.h(0)  # Hadamard gate
            qc.cx(0, 1)  # CNOT gate
            qc.measure_all()

            # Use basic simulator
            backend_sim = BasicSimulator()
            transpiled_qc = transpile(qc, backend_sim)

            # Run simulation
            job = backend_sim.run(transpiled_qc, shots=1024)
            result = job.result()
            counts = result.get_counts()

            duration = time.time() - start_time
            memory_end = psutil.Process().memory_info().rss / (1024 * 1024)
            cpu_end = psutil.cpu_percent(interval=None)

            return BenchmarkResult(
                backend=backend.value,
                operation="quantum_simulation",
                duration_seconds=duration,
                memory_usage_mb=memory_end - memory_start,
                cpu_usage_percent=cpu_end,
                success=True,
                additional_metrics={
                    "qubits_used": 2,
                    "shots": 1024,
                    "measurement_results": counts,
                    "circuit_depth": qc.depth()
                }
            )

        except ImportError:
            return BenchmarkResult(
                backend=backend.value,
                operation="quantum_simulation",
                duration_seconds=time.time() - start_time,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                success=False,
                error_message="Qiskit not installed"
            )
        except Exception as e:
            return BenchmarkResult(
                backend=backend.value,
                operation="quantum_simulation",
                duration_seconds=time.time() - start_time,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                success=False,
                error_message=str(e)
            )

    def test_inference(self, backend: BackendType, model_path: str, test_queries: List[str]) -> BenchmarkResult:
        """Test model inference performance."""
        from run_model import ThinkingModelInterface

        start_time = time.time()
        memory_start = psutil.Process().memory_info().rss / (1024 * 1024)
        cpu_start = psutil.cpu_percent(interval=None)

        try:
            # Load model
            engine = ThinkingModelInterface()
            engine.load_model(model_path)

            # Run inference on test queries
            responses = []
            for query in test_queries:
                response = engine.think(query)
                responses.append(response)

            duration = time.time() - start_time
            memory_end = psutil.Process().memory_info().rss / (1024 * 1024)
            cpu_end = psutil.cpu_percent(interval=None)

            return BenchmarkResult(
                backend=backend.value,
                operation="inference",
                duration_seconds=duration,
                memory_usage_mb=memory_end - memory_start,
                cpu_usage_percent=cpu_end,
                success=True,
                additional_metrics={
                    "queries_processed": len(test_queries),
                    "avg_response_length": sum(len(r) for r in responses) / len(responses),
                    "responses_generated": len(responses)
                }
            )

        except Exception as e:
            return BenchmarkResult(
                backend=backend.value,
                operation="inference",
                duration_seconds=time.time() - start_time,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                success=False,
                error_message=str(e)
            )

    def run_comprehensive_test(self, dataset_path: str = None, model_path: str = None) -> Dict[str, Any]:
        """Run comprehensive tests across all available backends."""
        print("Thinking Engine - Multi-Platform Testing Framework")
        print("=" * 60)
        print(f"Platform: {self.platform_info.os} {self.platform_info.architecture}")
        print(f"Python: {self.platform_info.python_version}")
        print(f"Available Backends: {', '.join(self.platform_info.available_backends)}")
        print(f"Hardware: {self.platform_info.hardware_info}")
        print()

        # Test data
        test_texts = [
            "Quantum computing uses quantum mechanics principles.",
            "Machine learning algorithms learn from data patterns.",
            "Neural networks consist of interconnected nodes called neurons.",
            "Sparse representations use efficient memory storage.",
            "Cognitive architectures model human thinking processes."
        ]

        test_queries = [
            "What is quantum computing?",
            "Explain machine learning",
            "How do neural networks work?"
        ]

        # Use default dataset if not provided
        if not dataset_path:
            dataset_path = "Introduction_to_Quantum_Computers_dataset"
        if not model_path:
            model_path = "thinking_model.think"  # Correct path to the models directory

        all_results = []

        # Test each available backend
        for backend_str in self.platform_info.available_backends:
            backend = BackendType(backend_str)
            print(f"Testing {backend.value.upper()} Backend")
            print("-" * 40)

            # Test sparse encoding
            print("Testing sparse encoding...")
            result = self.test_sparse_encoding(backend, test_texts)
            all_results.append(result)
            status = "PASSED" if result.success else "FAILED"
            print(f"   Duration: {result.duration_seconds:.2f}s, Memory: {result.memory_usage_mb:.1f}MB")
            if not result.success:
                print(f"   Error: {result.error_message}")

            # Test model training (skip for quantum backend)
            if backend != BackendType.QUANTUM_IBM and os.path.exists(dataset_path):
                print("Testing model training...")
                result = self.test_model_training(backend, dataset_path)
                all_results.append(result)
                status = "PASSED" if result.success else "FAILED"
                print(f"   Duration: {result.duration_seconds:.2f}s, Memory: {result.memory_usage_mb:.1f}MB")
                if not result.success:
                    print(f"   Error: {result.error_message}")

            # Test quantum simulation (only for quantum backend)
            if backend == BackendType.QUANTUM_IBM:
                print("Testing quantum simulation...")
                result = self.test_quantum_simulation(backend)
                all_results.append(result)
                status = "PASSED" if result.success else "FAILED"
                print(f"   Duration: {result.duration_seconds:.2f}s, Memory: {result.memory_usage_mb:.1f}MB")
                if not result.success:
                    print(f"   Error: {result.error_message}")

            # Test inference (skip for quantum backend)
            if backend != BackendType.QUANTUM_IBM and os.path.exists(model_path):
                print("Testing inference...")
                result = self.test_inference(backend, model_path, test_queries)
                all_results.append(result)
                status = "PASSED" if result.success else "FAILED"
                print(f"   Duration: {result.duration_seconds:.2f}s, Memory: {result.memory_usage_mb:.1f}MB")
                if not result.success:
                    print(f"   Error: {result.error_message}")

            print()

        # Generate comprehensive report
        report = self._generate_report(all_results)
        return report

    def _generate_report(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        report = {
            "platform_info": asdict(self.platform_info),
            "timestamp": time.time(),
            "total_tests": len(results),
            "passed_tests": len([r for r in results if r.success]),
            "failed_tests": len([r for r in results if not r.success]),
            "results": [asdict(r) for r in results],
            "performance_summary": {},
            "recommendations": []
        }

        # Performance summary by backend
        backend_performance = {}
        for result in results:
            if result.success:
                if result.backend not in backend_performance:
                    backend_performance[result.backend] = {
                        "total_time": 0,
                        "total_memory": 0,
                        "operations": []
                    }
                backend_performance[result.backend]["total_time"] += result.duration_seconds
                backend_performance[result.backend]["total_memory"] += result.memory_usage_mb
                backend_performance[result.backend]["operations"].append(result.operation)

        report["performance_summary"] = backend_performance

        # Generate recommendations
        if BackendType.CUDA.value in self.platform_info.available_backends:
            report["recommendations"].append("NVIDIA GPU available - consider using CUDA for accelerated training")
        if BackendType.MPS.value in self.platform_info.available_backends:
            report["recommendations"].append("Apple Silicon MPS available - use Metal Performance Shaders for optimal performance")
        if BackendType.QUANTUM_IBM.value in self.platform_info.available_backends:
            report["recommendations"].append("IBM Quantum simulator available - test quantum algorithms and circuits")

        return report

    def save_report(self, report: Dict[str, Any], filename: str = "multiplatform_test_report.json"):
        """Save test report to file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to: {filename}")

    def generate_training_plots(self, results: List[Dict[str, Any]], save_path: str = "training_comparison.png"):
        """Generate comparison plots for training progress across backends with system info."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt

            # Collect training data from results
            training_data = {}
            system_info = {}
            
            for result in results:
                if result.get("operation") == "model_training" and result.get("success", False):
                    metrics = result.get("additional_metrics")
                    if metrics and "training_progress" in metrics:
                        progress = metrics["training_progress"]
                        backend = result["backend"]
                        training_data[backend] = {
                            "epochs": progress["epochs"],
                            "loss": progress["loss_values"],
                            "accuracy": progress["accuracy_values"]
                        }
                        # Collect system info for each backend
                        system_info[backend] = {
                            "backend_type": backend,
                            "execution_time": result.get("duration_seconds", 0),
                            "memory_usage": result.get("memory_usage_mb", 0),
                            "cpu_usage": result.get("cpu_usage_percent", 0),
                            "total_epochs": len(progress["epochs"]),
                            "final_loss": progress["loss_values"][-1] if progress["loss_values"] else 0,
                            "final_accuracy": progress["accuracy_values"][-1] if progress["accuracy_values"] else 0
                        }

            if not training_data:
                print("No training data available for plotting")
                return

            # Create figure with subplots for training curves and system info
            fig = plt.figure(figsize=(20, 12))
            
            # Create subplots: 2 rows, 2 columns
            # Top row: Loss and Accuracy plots
            # Bottom row: System information table
            gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
            
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, :])  # Spans both columns for system info

            # Colors for different backends
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            backend_names = {
                'cpu': 'CPU',
                'cuda': 'NVIDIA GPU',
                'mps': 'Apple Silicon MPS',
                'metal': 'Metal GPU',
                'quantum_ibm': 'IBM Quantum'
            }

            # Plot loss curves
            for i, (backend, data) in enumerate(training_data.items()):
                color = colors[i % len(colors)]
                label = backend_names.get(backend, backend.upper())
                ax1.plot(data["epochs"], data["loss"], color=color, marker='o',
                        linewidth=2, markersize=4, label=f'{label} Loss')

            ax1.set_title('Training Loss Comparison Across Backends', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot accuracy curves
            for i, (backend, data) in enumerate(training_data.items()):
                color = colors[i % len(colors)]
                label = backend_names.get(backend, backend.upper())
                ax2.plot(data["epochs"], data["accuracy"], color=color, marker='s',
                        linewidth=2, markersize=4, label=f'{label} Accuracy')

            ax2.set_title('Training Accuracy Comparison Across Backends', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Accuracy', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # System Information Section
            ax3.axis('off')
            
            # Prepare system info table data
            table_data = []
            headers = ['Backend', 'CPU Cores', 'Memory (GB)', 'Execution Time (s)', 'Epochs', 'Final Loss', 'Final Accuracy', 'Memory Usage (MB)']
            
            # Add system hardware info
            cpu_count = psutil.cpu_count()
            memory_gb = round(psutil.virtual_memory().total / (1024**3), 2)
            
            for backend, info in system_info.items():
                row = [
                    backend_names.get(backend, backend.upper()),
                    str(cpu_count),
                    f"{memory_gb}",
                    f"{info['execution_time']:.3f}",
                    str(info['total_epochs']),
                    f"{info['final_loss']:.4f}",
                    f"{info['final_accuracy']:.4f}",
                    f"{info['memory_usage']:.2f}"
                ]
                table_data.append(row)
            
            # Create table
            table = ax3.table(cellText=table_data,
                            colLabels=headers,
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.12, 0.1, 0.1, 0.12, 0.08, 0.1, 0.12, 0.12])
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style the table
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color code backend rows
            for i, backend in enumerate(system_info.keys()):
                color = colors[i % len(colors)]
                for j in range(len(headers)):
                    table[(i+1, j)].set_facecolor(color)
                    table[(i+1, j)].set_alpha(0.3)
            
            # Add platform info
            platform_text = f"Platform: {platform.system()} {platform.machine()} | Python: {platform.python_version()} | System Memory: {memory_gb}GB"
            ax3.text(0.5, 0.95, platform_text, transform=ax3.transAxes, 
                    ha='center', va='top', fontsize=12, fontweight='bold')

            # Overall title
            fig.suptitle('Thinking Engine Multi-Platform Training Comparison (1000 Epochs)',
                        fontsize=18, fontweight='bold', y=0.95)

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training comparison plot with system info saved to: {save_path}")

            # Generate individual plots for each backend
            plot_files = [save_path]
            
            for backend, data in training_data.items():
                # Create individual figure for this backend
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                backend_display_name = backend_names.get(backend, backend.upper())
                color = colors[list(training_data.keys()).index(backend) % len(colors)]

                # Plot loss curve for this backend
                ax1.plot(data["epochs"], data["loss"], color=color, marker='o',
                        linewidth=2, markersize=4, label=f'{backend_display_name} Loss')
                ax1.set_title(f'{backend_display_name} Training Loss', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Epoch', fontsize=12)
                ax1.set_ylabel('Loss', fontsize=12)
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Plot accuracy curve for this backend
                ax2.plot(data["epochs"], data["accuracy"], color=color, marker='s',
                        linewidth=2, markersize=4, label=f'{backend_display_name} Accuracy')
                ax2.set_title(f'{backend_display_name} Training Accuracy', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Epoch', fontsize=12)
                ax2.set_ylabel('Accuracy', fontsize=12)
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # Overall title
                fig.suptitle(f'Thinking Engine {backend_display_name} Training Progress (1000 Epochs)',
                            fontsize=16, fontweight='bold', y=0.98)

                plt.tight_layout()

                # Save individual plot
                individual_filename = f"training_{backend}_individual.png"
                plt.savefig(individual_filename, dpi=300, bbox_inches='tight')
                plot_files.append(individual_filename)
                print(f"Individual training plot for {backend_display_name} saved to: {individual_filename}")

                plt.close(fig)  # Close to free memory
            
            plt.close(fig)  # Close the main comparison plot

            print(f"\nGenerated {len(plot_files)} training plot files:")
            for plot_file in plot_files:
                print(f"  - {plot_file}")

            # Show plot if running interactively
            try:
                plt.show()
            except:
                pass  # May not work in headless environments

        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Error generating plots: {e}")

def main():
    """Main testing function."""
    tester = MultiPlatformTester()

    # Run comprehensive tests
    report = tester.run_comprehensive_test()

    # Generate training comparison plots
    print("\nGenerating training comparison plots...")
    tester.generate_training_plots(report["results"])

    # Save report
    tester.save_report(report)

    # Print summary
    print("\nMulti-Platform Testing Complete!")
    print(f"Total Tests: {report['total_tests']}")
    print(f"Passed: {report['passed_tests']}")
    print(f"Failed: {report['failed_tests']}")
    print("\nPerformance Summary:")
    for backend, perf in report['performance_summary'].items():
        print(f"  {backend.upper()}: {perf['total_time']:.2f}s total, {len(perf['operations'])} operations")

    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")

if __name__ == "__main__":
    main()
