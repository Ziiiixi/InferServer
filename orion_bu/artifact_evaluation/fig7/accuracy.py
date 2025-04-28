import re

# Path to your log file
log_path = '123.log'

# Patterns
colocation_predict_pattern = re.compile(r'The predict latency of co-location kernels (\d+) and (\d+) is: (\d+), the real duration took (\d+) nanoseconds')
single_predict_pattern = re.compile(r'using only single profile, the predict latency is: (\d+)')

results = []

def compute_accuracy(predicted, real, penalty_factor=1.5):
    error = abs(predicted - real)
    if predicted >= real:
        return max(0, 1 - error / real)
    else:
        return max(0, 1 - penalty_factor * error / real)

with open(log_path, 'r') as file:
    lines = file.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]

        match_colocation = colocation_predict_pattern.search(line)
        if match_colocation:
            kernel1, kernel2, pred_colocation, real_duration = match_colocation.groups()
            pred_colocation = int(pred_colocation)
            real_duration = int(real_duration)

            if i + 1 < len(lines):
                match_single = single_predict_pattern.search(lines[i + 1])
                if match_single:
                    pred_single = int(match_single.group(1))

                    acc_colocation = compute_accuracy(pred_colocation, real_duration)
                    acc_single = compute_accuracy(pred_single, real_duration)

                    results.append({
                        'kernels': f'{kernel1}-{kernel2}',
                        'real_duration': real_duration,
                        'pred_colocation': pred_colocation,
                        'acc_colocation': acc_colocation,
                        'pred_single': pred_single,
                        'acc_single': acc_single
                    })

                    i += 1  # skip next line

        i += 1

# Print results
for r in results:
    print(f"Kernels {r['kernels']}:")
    print(f"  Real: {r['real_duration']} ns")
    print(f"  Co-location Predict: {r['pred_colocation']} ns, Accuracy: {r['acc_colocation']:.4f}")
    print(f"  Single Predict:      {r['pred_single']} ns, Accuracy: {r['acc_single']:.4f}")
    print()

# Summary
if results:
    avg_colocation_acc = sum(r['acc_colocation'] for r in results) / len(results)
    avg_single_acc = sum(r['acc_single'] for r in results) / len(results)
    print(f"Average Accuracy (with soft penalty on underestimation) - Co-location: {avg_colocation_acc:.4f}")
    print(f"Average Accuracy (with soft penalty on underestimation) - Single Profile: {avg_single_acc:.4f}")
