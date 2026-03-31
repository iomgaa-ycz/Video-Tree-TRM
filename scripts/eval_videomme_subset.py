import json
import os
import time
from pathlib import Path
from datetime import datetime
import torch

from video_tree_trm.config import Config
from video_tree_trm.pipeline import Pipeline
from utils.logger_system import log_msg

def parse_option(pred: str) -> str:
    """从 VLM 输出中提取选项字母 (A/B/C/D)。"""
    pred = pred.strip().upper()
    # 常见情况：开头就是 A/B/C/D
    if len(pred) > 0 and pred[0] in "ABCD":
        # 排除 "A statement..." 这种误判，检查后面是否跟了点或空格
        if len(pred) == 1 or pred[1] in ". )":
            return pred[0]
    
    # 模糊匹配：查找出现的第一个选项
    for char in pred:
        if char in "ABCD":
            return char
    return "N/A"

def main():
    # 1. 加载配置
    cfg_path = "config/videomme.yaml"
    cfg = Config.load(cfg_path)
    
    # 2. 初始化 Pipeline
    pipeline = Pipeline(cfg)
    
    # 3. 加载评测数据
    eval_jsonl = "data/videomme/queries/sample_eval.jsonl"
    if not os.path.exists(eval_jsonl):
        print(f"评测文件不存在: {eval_jsonl}")
        return

    with open(eval_jsonl, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    print(f"开始评测，共 {len(samples)} 条样本...")
    
    results = []
    correct = 0
    total = 0
    
    # 准备输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"tests/outputs/videomme_eval_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log_md_path = out_dir / "summary.md"
    with open(log_md_path, "w", encoding="utf-8") as f_sum:
        f_sum.write(f"# VideoMME 子集评测汇总 ({timestamp})\n\n")
        f_sum.write(f"- 配置文件: `{cfg_path}`\n")
        f_sum.write(f"- 样本总数: {len(samples)}\n\n")
        f_sum.write("| 样本ID | 视频ID | 预估 | 真值 | 结果 | 轮次 | 关键帧数 |\n")
        f_sum.write("|---|---|---|---|---|---|---|\n")

        for idx, sample in enumerate(samples):
            query = sample["query"]
            gt_answer = sample["answer"]
            source_path = sample["source_path"]
            q_id = sample.get("question_id", str(idx))
            yt_id = sample.get("youtube_id", "unknown")

            print(f"[{idx+1}/{len(samples)}] 正在处理 {q_id} (Video: {yt_id})...")
            
            try:
                # 记录开始时间
                start_time = time.time()
                
                # 执行 Pipeline
                # 注意：我们之前更新了 Pipeline.query 支持路径
                # 这里我们直接传 source_path
                pred_raw = pipeline.query(query, source_path, modality="video")
                pred_option = parse_option(pred_raw)
                
                is_correct = (pred_option == gt_answer)
                if is_correct:
                    correct += 1
                total += 1
                
                elapsed = time.time() - start_time
                
                # 记录单样本详细日志
                sample_log_path = out_dir / f"{q_id}.md"
                with open(sample_log_path, "w", encoding="utf-8") as f_sample:
                    f_sum_line = f"# 评测样本: {q_id}\n\n"
                    f_sum_line += f"- **视频 ID**: {yt_id}\n"
                    f_sum_line += f"- **问题**: {query}\n"
                    f_sum_line += f"- **选项**: {sample.get('options')}\n"
                    f_sum_line += f"- **真值**: {gt_answer}\n"
                    f_sum_line += f"- **预测**: {pred_option} (原文: {pred_raw})\n"
                    f_sum_line += f"- **耗时**: {elapsed:.2f}s\n\n"
                    f_sample.write(f_sum_line)
                
                # 写入汇总表
                res_str = "✅" if is_correct else "❌"
                f_sum.write(f"| {q_id} | {yt_id} | {pred_option} | {gt_answer} | {res_str} | - | - |\n")
                f_sum.flush()
                
            except Exception as e:
                print(f"样本 {q_id} 处理失败: {e}")
                f_sum.write(f"| {q_id} | {yt_id} | ERROR | {gt_answer} | ⚠️ | - | - |\n")
                continue

        # 最终汇总
        accuracy = correct / total if total > 0 else 0
        summary_footer = f"\n## 最终统计\n- **总计**: {total}\n- **正确**: {correct}\n- **准确率**: {accuracy:.2%} \n"
        f_sum.write(summary_footer)
        print(summary_footer)

if __name__ == "__main__":
    main()
