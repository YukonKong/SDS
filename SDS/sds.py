import hydra
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
import os
import re
import subprocess
from pathlib import Path
import shutil
import torch
from utils.misc import *
from utils.extract_task_code import *
from utils.vid_utils import create_grid_image, encode_image, save_grid_image
from utils.easy_vit_pose import vitpose_inference
import cv2
import openai
import os
from agents import SUSGenerator

# 忽略警告，保持输出的整洁
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# 设置根目录、API密钥
SDS_ROOT_DIR = os.getcwd()
ROOT_DIR = f"{SDS_ROOT_DIR}/.."
openai.api_key = os.getenv("OPENAI_API_KEY")


# 使用hydra库来加载配置文件
@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    # 日志记录配置信息
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {SDS_ROOT_DIR}")
    logging.info(f"Running for {cfg.iteration} iterations")
    logging.info(f"Training each RF for: {cfg.train_iterations} iterations")
    logging.info(f"Generating {cfg.sample} reward function samples per iteration")

    # 加载LLM模型和任务信息
    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info(f"Imitation Task: {cfg.task.description}")

    # 设置环境名称
    env_name = cfg.env_name.lower()

    # 读取任务奖励函数文件和观测函数文件
    task_rew_file = f"{ROOT_DIR}/{env_name}/{cfg.reward_template_file}"
    task_obs_file = f"{SDS_ROOT_DIR}/envs/{env_name}.py"
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_rew_code_string = file_to_string(task_rew_file)
    task_obs_code_string = file_to_string(task_obs_file)
    output_file = f"{ROOT_DIR}/{env_name}/{cfg.reward_output_file}"

    # 加载所有提示词文本
    prompt_dir = f"{SDS_ROOT_DIR}/prompts"
    initial_reward_engineer_system = file_to_string(
        f"{prompt_dir}/initial_reward_engineer_system.txt"
    )
    code_output_tip = file_to_string(f"{prompt_dir}/code_output_tip.txt")
    code_feedback = file_to_string(f"{prompt_dir}/code_feedback.txt")
    initial_reward_engineer_user = file_to_string(
        f"{prompt_dir}/initial_reward_engineer_user.txt"
    )
    reward_signature = file_to_string(f"{prompt_dir}/reward_signatures/{env_name}.txt")
    policy_feedback = file_to_string(f"{prompt_dir}/policy_feedback.txt")
    execution_error_feedback = file_to_string(
        f"{prompt_dir}/execution_error_feedback.txt"
    )
    initial_task_evaluator_system = file_to_string(
        f"{prompt_dir}/initial_task_evaluator_system.txt"
    )

    # 处理演示视频，生成帧网格
    demo_video_name = cfg.task.video
    video_do_crop = cfg.task.crop
    logging.info(
        f"Demonstration Video: {demo_video_name}, Crop Option: {cfg.task.crop_option}"
    )
    gt_frame_grid = create_grid_image(
        f"{ROOT_DIR}/videos/{demo_video_name}",
        grid_size=(cfg.task.grid_size, cfg.task.grid_size),
        crop=video_do_crop,
        crop_option=cfg.task.crop_option,
    )
    save_grid_image(gt_frame_grid, "gt_demo.png")

    # 使用VITPose进行姿态估计，生成带有姿态估计的帧网格
    annotated_video_path = vitpose_inference(
        f"{ROOT_DIR}/videos/{demo_video_name}",
        f"{workspace_dir}/pose-estimate/gt-pose-estimate",
    )
    gt_annotated_frame_grid = create_grid_image(
        annotated_video_path,
        grid_size=(cfg.task.grid_size, cfg.task.grid_size),
        crop=video_do_crop,
        crop_option=cfg.task.crop_option,
    )
    save_grid_image(gt_annotated_frame_grid, "gt_demo_annotated.png")

    # 设置评估脚本的路径
    eval_script_dir = os.path.join(ROOT_DIR, "forward_locomotion_sds/scripts/play.py")

    # 将生成的网格图像编码
    encoded_gt_frame_grid = encode_image(f"{workspace_dir}/gt_demo.png")

    # 生成SUS提示词
    sus_generator = SUSGenerator(cfg, prompt_dir)
    SUS_prompt = sus_generator.generate_sus_prompt(encoded_gt_frame_grid)

    # 生成初始奖励工程师系统提示词
    initial_reward_engineer_system = (
        initial_reward_engineer_system.format(
            task_reward_signature_string=reward_signature,
            task_obs_code_string=task_obs_code_string,
        )
        + code_output_tip
    )

    # 生成初始奖励工程师用户提示词
    initial_reward_engineer_user = initial_reward_engineer_user.format(
        sus_string=SUS_prompt, task_obs_code_string=task_obs_code_string
    )

    # 生成初始任务评估系统提示词
    initial_task_evaluator_system = initial_task_evaluator_system.format(
        sus_string=SUS_prompt
    )

    # 生成奖励查询消息
    reward_query_messages = [
        {"role": "system", "content": initial_reward_engineer_system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": initial_reward_engineer_user},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_gt_frame_grid}",
                        "detail": cfg.image_quality,
                    },
                },
            ],
        },
    ]

    # 创建工作目录
    os.mkdir(f"{workspace_dir}/training_footage")
    os.mkdir(f"{workspace_dir}/contact_sequence")

    # 初始化变量
    DUMMY_FAILURE = -10000.0
    max_successes = []
    max_successes_reward_correlation = []
    # execute_rates = []
    best_code_paths = []
    max_reward_code_path = None

    best_footage = None
    best_contact = None

    # 开始迭代，生成和优化奖励函数。
    for iter in range(cfg.iteration):

        # 日志记录当前迭代信息
        logging.info(
            f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}"
        )

        # 向GPT询问并获得回答
        responses, prompt_tokens, total_completion_token, total_token = gpt_query(
            cfg.sample, reward_query_messages, cfg.temperature, cfg.model
        )

        # 如果cfg.sample为1，则输出GPT的回答
        if cfg.sample == 1:
            logging.info(
                f"Iteration {iter}: GPT Output:\n "
                + responses[0]["message"]["content"]
                + "\n"
            )

        # 日志记录令牌信息
        logging.info(
            f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}"
        )

        # 初始化变量
        code_runs = []
        rl_runs = []
        footage_grids_dir = []
        contact_pattern_dirs = []

        successful_runs_index = []

        eval_success = False

        # 对每个回答进行处理
        for response_id in range(cfg.sample):
            response_cur = responses[response_id]["message"]["content"]
            # print(response_cur)

            # 日志记录当前回答的ID
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # 正则表达式提取GPT响应中包含的python代码
            patterns = [
                r"```python(.*?)```",
                r"```(.*?)```",
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string

            # 删除不必要的import语句
            lines = code_string.split("\n")
            lines = [" " * 4 + line for line in lines]
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])
                    break

            # 调整代码字符串的缩进，确保每一行都比第一行增加4个空格的缩进
            def ensure_doubly_indented(code_str):
                lines = code_str.splitlines()

                base_indentation = len(lines[0]) - len(lines[0].lstrip())

                def adjust_indentation(line):
                    stripped_line = line.lstrip()
                    current_indentation = len(line) - len(stripped_line)

                    return (
                        " " * (4 + current_indentation - base_indentation)
                        + stripped_line
                    )

                adjusted_lines = []
                for i, line in enumerate(lines):
                    if i == 0 and line.strip().startswith("def"):
                        adjusted_lines.append(adjust_indentation(line))
                    else:
                        adjusted_lines.append(adjust_indentation(line))

                # 用换行符将这些行重新组合成一个字符串
                adjusted_code = "\n".join(adjusted_lines)

                return adjusted_code

            code_string = ensure_doubly_indented(code_string)
            code_runs.append(code_string)

            # 添加SDS奖励签名到环境代码中
            cur_task_rew_code_string = task_rew_code_string.replace(
                "# INSERT SDS REWARD HERE", code_string
            )

            # 当输出包含有效的代码字符串时，保存新的环境代码
            with open(output_file, "w") as file:
                file.writelines(cur_task_rew_code_string + "\n")

            with open(
                f"env_iter{iter}_response{response_id}_rewardonly.py", "w"
            ) as file:
                file.writelines(code_string + "\n")

            # 复制生成的环境代码到hydra输出目录以备后续使用
            shutil.copy(output_file, f"env_iter{iter}_response{response_id}.py")

            # 寻找空闲的GPU以加速RL
            set_freest_gpu()

            # 携带参数执行python脚本
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            with open(rl_filepath, "w") as f:
                # 构建RL训练命令
                command = f"python -u {ROOT_DIR}/{env_name}/{cfg.train_script} --iterations {cfg.train_iterations} --dr-config off --reward-config sds --no-wandb"
                command = command.split(" ")
                # 执行命令并记录输出
                process = subprocess.run(command, stdout=f, stderr=f)
            # 检查训练是否成功
            training_success = block_until_training(
                rl_filepath,
                success_keyword=cfg.success_keyword,
                failure_keyword=cfg.failure_keyword,
                log_status=True,
                iter_num=iter,
                response_id=response_id,
            )
            # 记录进程
            rl_runs.append(process)

            # 训练成功则执行评估脚本，保存相关数据
            if training_success:
                # 设置路径
                training_log_dir = extract_training_log_dir(rl_filepath)
                full_training_log_dir = os.path.join(
                    f"{ROOT_DIR}/{env_name}/runs/", training_log_dir
                )
                contact_pattern_dir = os.path.join(
                    full_training_log_dir, "contact_sequence.png"
                )
                # 构建评估脚本命令
                eval_script = f"python -u {eval_script_dir} --run {full_training_log_dir} --dr-config sds --headless --save_contact"
                training_footage_dir = os.path.join(full_training_log_dir, "videos")

                try:
                    # 执行评估脚本
                    subprocess.run(eval_script.split(" "))

                    annotated_video_path = vitpose_inference(
                        os.path.join(training_footage_dir, "play.mp4"),
                        f"{workspace_dir}/pose-estimate/sample-pose-estimate",
                    )
                    # 创建训练帧网格图像
                    training_frame_grid = create_grid_image(
                        annotated_video_path, training_fixed_length=True
                    )
                    # save_grid_image(training_annotated_frame_grid,f"training_footage/training_frame_{iter}_{response_id}_annotated.png")

                    footage_grid_save_dir = (
                        f"training_footage/training_frame_{iter}_{response_id}.png"
                    )
                    save_grid_image(training_frame_grid, footage_grid_save_dir)

                    contact_sequence_save_dir = f"{workspace_dir}/contact_sequence/contact_sequence_{iter}_{response_id}.png"

                    shutil.copy(contact_pattern_dir, contact_sequence_save_dir)

                    footage_grids_dir.append(footage_grid_save_dir)

                    contact_pattern_dirs.append(contact_sequence_save_dir)

                    successful_runs_index.append(response_id)

                    eval_success = True

                except:
                    # 处理评估失败的情况，由于奖励运行时间错误，没有保存任何视频
                    logging.info(
                        f"Iteration {iter}: Code Run {response_id} Failed to Evaluate, Not evaluated"
                    )
            # 训练失败则记录错误信息
            else:
                logging.info(
                    f"Iteration {iter}: Code Run {response_id} Unstable, Not evaluated"
                )

        # 如果代码评估失败了，则重复迭代
        if not eval_success and cfg.sample != 1:
            # execute_rates.append(0.)
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info(
                "All code evaluation failed! Repeat this iteration from the current message checkpoint!"
            )
            continue

        code_feedbacks = []
        contents = []
        reward_correlations = []
        code_paths = []

        # 初始化执行成功标志为False
        exec_success = False
        for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            code_paths.append(f"env_iter{iter}_response{response_id}.py")
            try:
                # 打开rl_filepath文件，读取内容
                with open(rl_filepath, "r") as f:
                    stdout_str = f.read()
            except:
                # 如果打开文件失败，则构造错误反馈内容
                content = execution_error_feedback.format(
                    traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!"
                )
                content += code_output_tip
                # 将错误反馈内容添加到contents列表中
                contents.append(content)
                reward_correlations.append(DUMMY_FAILURE)
                continue

            content = ""
            traceback_msg = filter_traceback(stdout_str)

            # 如果代码运行成功
            if traceback_msg == "":
                # 如果RL执行没有错误，则提供策略统计反馈
                exec_success = True
                run_log = construct_run_log(stdout_str)

                # 获取训练迭代次数、计算每10个epoch的频率
                train_iterations = np.array(run_log["iterations/"]).shape[0]
                epoch_freq = max(int(train_iterations // 10), 1)

                epochs_per_log = 10
                # 添加反馈内容
                content += policy_feedback.format(
                    epoch_freq=epochs_per_log * epoch_freq
                )

                # 计算人工构造的奖励和GPT奖励之间的相关性
                if "gt_reward" in run_log and "gpt_reward" in run_log:
                    gt_reward = np.array(run_log["gt_reward"])
                    gpt_reward = np.array(run_log["gpt_reward"])
                    reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                    reward_correlations.append(reward_correlation)

                # 将奖励组件日志添加到反馈中
                for metric in sorted(run_log.keys()):
                    if "/" not in metric:
                        metric_cur = [
                            "{:.2f}".format(x) for x in run_log[metric][::epoch_freq]
                        ]
                        metric_cur_max = max(run_log[metric])
                        metric_cur_mean = sum(run_log[metric]) / len(run_log[metric])

                        metric_cur_min = min(run_log[metric])
                        if metric != "gt_reward" and metric != "gpt_reward":
                            metric_name = metric
                            content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"

                # 添加代码反馈
                code_feedbacks.append(code_feedback)
                content += code_feedback
            else:
                # 否则，提供执行回溯错误反馈
                reward_correlations.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            content += code_output_tip
            contents.append(content)

        # 如果代码生成失败，则进行日志记录
        if not exec_success and cfg.sample != 1:
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info(
                "All code generation failed! Repeat this iteration from the current message checkpoint!"
            )
            continue

        def compute_similarity_score_gpt(footage_grids_dir, contact_pattern_dirs):

            # 定义评估器查询内容
            evaluator_query_content = [
                {"type": "text", "text": "You will be rating the following images:"}
            ]

            # 遍历视频目录
            for footage_dir in footage_grids_dir:

                # 编码视频
                encoded_footage = encode_image(footage_dir)

                # 将编码后的视频添加到评估器查询内容中
                evaluator_query_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_footage}",
                            "detail": cfg.image_quality,
                        },
                    }
                )

            # 定义接触模式评估器查询内容
            contact_evaluator_query_content = [
                {
                    "type": "text",
                    "text": "They have the following corresponding foot contact sequence plots, where FR means Front Right Foot, FL means Front Left Foot, RR means Rear Right Foot and RL means Rear Right Foot",
                }
            ]

            # 遍历接触模式目录
            for contact_dir in contact_pattern_dirs:

                # 编码接触模式
                encoded_contact = encode_image(contact_dir)

                # 将编码后的接触模式添加到接触模式评估器查询内容中
                contact_evaluator_query_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_contact}",
                            "detail": cfg.image_quality,
                        },
                    }
                )

            # 如果有最佳视频，则将其添加到评估器查询内容中
            if best_footage is not None:
                evaluator_query_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{best_footage}",
                            "detail": cfg.image_quality,
                        },
                    }
                )

                contact_evaluator_query_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{best_contact}",
                            "detail": cfg.image_quality,
                        },
                    }
                )

                successful_runs_index.append(-1)

            # 如果使用注释，则编码注释后的帧网格
            if cfg.task.use_annotation:
                encoded_gt_frame_grid = encode_image(
                    f"{workspace_dir}/gt_demo_annotated.png"
                )
            else:
                # 否则编码原始帧网格
                encoded_gt_frame_grid = encode_image(f"{workspace_dir}/gt_demo.png")

            # 定义评估器查询消息
            evaluator_query_messages = [
                {"role": "system", "content": initial_task_evaluator_system},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here is the image demonstrating the ground truth task",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_gt_frame_grid}",
                                "detail": cfg.image_quality,
                            },
                        },
                    ],
                },
                None,
                None,
            ]

            # 将评估器查询内容和接触模式评估器查询内容添加到评估器查询消息中
            evaluator_query_messages[2] = {
                "role": "user",
                "content": evaluator_query_content,
            }

            evaluator_query_messages[3] = {
                "role": "user",
                "content": contact_evaluator_query_content,
            }

            # 打印评估信息
            logging.info("Evaluating...")
            # 使用GPT查询评估器查询消息
            eval_responses, _, _, _ = gpt_query(
                1, evaluator_query_messages, cfg.temperature, cfg.model
            )

            # 获取评估响应
            eval_responses = eval_responses[0]["message"]["content"]

            # 使用正则表达式获取分数
            scores_re = re.findall(r"\[([^\]]*)\](?!.*\[)", eval_responses)
            scores_re = scores_re[-1]

            # 将分数转换为浮点数
            scores = [float(x) for x in scores_re.split(",")]

            # 如果只有一个分数，则返回0和True
            if len(scores) == 1:
                logging.info(f"Best Sample Index: {0}")
                return 0, True
            else:
                # 否则获取最佳和第二好的样本索引
                best_idx_in_successful_runs = np.argmax(scores)
                second_best_idx_in_successful_runs = np.argsort(scores)[-2]

                best_idx = successful_runs_index[best_idx_in_successful_runs]
                second_best_idx = successful_runs_index[
                    second_best_idx_in_successful_runs
                ]
                # logging.info(f"Iteration {iter}: Prompt Tokens: {eval_prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")

                # 打印最佳和第二好的样本索引
                logging.info(
                    f"Best Sample Index: {best_idx}, Second Best Sample Index: {second_best_idx}"
                )

                # 将评估器查询消息和评估响应保存到文件中
                with open(f"evaluator_query_messages_{iter}.json", "w") as file:
                    json.dump(
                        evaluator_query_messages
                        + [{"role": "assistant", "content": eval_responses}],
                        file,
                        indent=4,
                    )

                # 如果最佳样本是之前的最佳视频，则返回第二好的样本索引和False
                if best_idx == -1:
                    # Best sample is the previous best footage
                    return second_best_idx, False

                # 否则返回最佳样本索引和True
                return best_idx, True

        # 计算GPT的相似度得分
        best_sample_idx, improved = compute_similarity_score_gpt(
            footage_grids_dir, contact_pattern_dirs
        )

        # 获取最佳样本的内容
        best_content = contents[best_sample_idx]

        # 如果improved为True，则记录日志，表示已经生成了更好的奖励函数
        if improved:
            logging.info(
                f"Iteration {iter}: A better reward function has been generated"
            )
            # 获取最佳样本的代码路径
            max_reward_code_path = code_paths[best_sample_idx]
            # 对最佳样本进行编码
            best_footage = encode_image(
                f"{workspace_dir}/training_footage/training_frame_{iter}_{best_sample_idx}.png"
            )

            # 获取rl文件路径
            rl_filepath = f"env_iter{iter}_response{best_sample_idx}.txt"
            # 提取训练日志目录
            training_log_dir = extract_training_log_dir(rl_filepath)
            # 获取完整训练日志目录
            full_training_log_dir = os.path.join(
                f"{ROOT_DIR}/{env_name}/runs/", training_log_dir
            )
            # 获取接触模式目录
            contact_pattern_dir = os.path.join(
                full_training_log_dir, "contact_sequence.png"
            )

            # 对接触模式进行编码
            best_contact = encode_image(contact_pattern_dir)

        # 将最佳样本的代码路径添加到best_code_paths列表中
        best_code_paths.append(code_paths[best_sample_idx])

        # 记录当前迭代的最佳生成ID
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        # 记录当前迭代的GPT输出内容
        logging.info(
            f"Iteration {iter}: GPT Output Content:\n"
            + responses[best_sample_idx]["message"]["content"]
            + "\n"
        )
        # 记录当前迭代的用户内容
        logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")

        # 如果reward_query_messages的长度为2
        if len(reward_query_messages) == 2:
            # 将assistant的回复添加到reward_query_messages中
            reward_query_messages += [
                {
                    "role": "assistant",
                    "content": responses[best_sample_idx]["message"]["content"],
                }
            ]
            # 将best_content添加到reward_query_messages中
            reward_query_messages += [{"role": "user", "content": best_content}]
        # 否则，如果reward_query_messages的长度为4
        else:
            # 断言reward_query_messages的长度为4
            assert len(reward_query_messages) == 4
            # 将assistant的回复替换为best_sample_idx对应的回复
            reward_query_messages[-2] = {
                "role": "assistant",
                "content": responses[best_sample_idx]["message"]["content"],
            }
            # 将best_content替换为reward_query_messages的最后一个元素
            reward_query_messages[-1] = {"role": "user", "content": best_content}

        # 将字典保存为JSON文件
        with open("reward_query_messages.json", "w") as file:
            json.dump(reward_query_messages, file, indent=4)

    if max_reward_code_path is None:
        logging.info("All iterations of code generation failed, aborting...")
        logging.info(
            "Please double check the output env_iter*_response*.txt files for repeating errors!"
        )
        exit()
    logging.info(f"Best Reward Code Path: {max_reward_code_path}")

    # 将最佳奖励代码路径转换为字符串
    best_reward = file_to_string(max_reward_code_path)
    # 将最佳奖励代码写入输出文件
    with open(output_file, "w") as file:
        file.writelines(best_reward + "\n")

    # 获取最佳性能策略的运行目录
    with open(max_reward_code_path.replace(".py", ".txt"), "r") as file:
        lines = file.readlines()
    for line in lines:
        # 查找以"Dashboard: "开头行
        if line.startswith("Dashboard: "):
            # 获取运行目录
            run_dir = line.split(": ")[1].strip()
            # 将运行目录中的"http://app.dash.ml/"替换为"{ROOT_DIR}/{env_name}/runs/"
            run_dir = run_dir.replace(
                "http://app.dash.ml/", f"{ROOT_DIR}/{env_name}/runs/"
            )
            logging.info("Best policy run directory: " + run_dir)


if __name__ == "__main__":
    main()
