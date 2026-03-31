import json
import os
from pathlib import Path

# 1. 获取已建树视频的列表
tree_dir = Path("data/videomme/trees")
video_dir = Path("data/videomme/videos")
built_stems = set()
for f in tree_dir.glob("*_video.json"):
    stem = f.name.replace("_video.json", "")
    built_stems.add(stem)

print(f"找到 {len(built_stems)} 个已建好的视频树。")

# 2. 扫描视频目录，建立 youtube_id 到实际文件路径的映射
# 有些视频可能有后缀如 .f299.mp4
video_map = {}
for f in video_dir.glob("*.mp4"):
    full_stem = f.stem
    # 尝试提取 youtube_id (通常是第一个点之前的部分)
    yt_id = full_stem.split('.')[0]
    video_map[yt_id] = str(f)
    # 也存储完整的 stem 以备不时之需
    video_map[full_stem] = str(f)

# 3. 过滤 QA 数据
qa_path = Path("data/videomme/metadata/long_videos_qa.jsonl")
output_path = Path("data/videomme/queries/sample_eval.jsonl")

samples = []
found_ids = set()
with open(qa_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        yt_id = data.get("youtube_id")
        
        # 检查该视频是否已建树
        # 匹配逻辑：yt_id 必须在 built_stems 中，或者 stem 匹配
        match_stem = None
        if yt_id in built_stems:
            match_stem = yt_id
        else:
            # 模糊匹配：查找包含 yt_id 的 built_stem
            for s in built_stems:
                if s.startswith(yt_id):
                    match_stem = s
                    break
        
        if match_stem:
            # 获取 source_path
            # 如果 video_map 里有完整匹配，用它；否则拼凑
            source_path = video_map.get(match_stem, f"data/videomme/videos/{match_stem}.mp4")
            
            # 构建 sample
            sample = {
                "query": data.get("question"),
                "answer": data.get("answer"),
                "options": data.get("options"),
                "source_path": source_path,
                "modality": "video",
                "timestamp": 0.0,  # 原始标注中未发现具体 timestamp，先填 0.0 作为占位
                "question_id": data.get("question_id"),
                "youtube_id": yt_id
            }
            samples.append(sample)
            found_ids.add(yt_id)

print(f"从 {len(found_ids)} 个视频中提取了 {len(samples)} 条 QA 样本。")

# 4. 写入文件
with open(output_path, 'w', encoding='utf-8') as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')

print(f"样本子集已保存至: {output_path}")
