# 随机输出9,10：追加seed104、105

只追加原random_operator设置的两个seed，不更换组合、训练目标或模型。数据生成、模型和训练程序直接使用父目录原文件，校验和与原seed101–103相同。每轮27万真实算术、13.5万检索、18万随机输出任务，固定各1200轮。随机结果每轮独立刷新。

新seed同时改变初始化、数据采样和随机标签，与原三个seed口径相同。只运行random_operator，不额外运行heldout_pairs。共享GPU1、6；screen qt1/randomop_104105；关闭VS Code不影响训练。

启动：bash start.sh。五seed实时表在父目录random_operator_5seeds_results.md；新训练日志/检查点仍在父目录runs/random_operator_seed104与105。protocol_manifest.json记录设置和原始文件校验和，validation.json记录数据留出和标签刷新检查。原三seed结果与检查点不覆盖，五个seed全部按固定第1200轮汇总，结束后自动核对所有保存预测。
