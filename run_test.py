import os
from multiprocessing import Process

def fun1(command):
    os.system(command)


if __name__ == '__main__':
    # 使用对应数据集的对应模型来测试不同的方法
    process_list = []
    data_list = ["zhongyin_data", "0th_ts_train", "15th_ts_train", "16th_ts_train", "17th_ts_train", "18th_ts_train", "19th_ts_train", "24th_ts_train", "apascal", "bank-additional-full_normalised", "probe", "creditcard"]

    criterion_distance = "--criterion distance"

    criterion_lof = "--criterion lof"

    criterion_iforest = "--criterion iforest"

    out_c_30 = "--out-c 30 "

    out_c_20 = "--out-c 20 "

    docker_command_neither = "CUDA_VISIBLE_DEVICES={GPU} python test.py " \
                     "--data-path data/{test_data}.csv " \
                     "--load-path save_model_{data}/neither/ " \
                     "--log-path test_logs/{data}/neither.log "  

    docker_command_pairwise = "CUDA_VISIBLE_DEVICES={GPU} python test.py " \
                     "--data-path data/{test_data}.csv " \
                     "--load-path save_model_{data}/pairwise/ " \
                     "--log-path test_logs/{data}/pairwise.log --use-pairwise " 

    docker_command_momentum = "CUDA_VISIBLE_DEVICES={GPU} python test.py " \
                    "--data-path data/{test_data}.csv " \
                    "--load-path save_model_{data}/momentum/ " \
                    "--log-path test_logs/{data}/momentum.log " 

    docker_command_both = "CUDA_VISIBLE_DEVICES={GPU} python test.py " \
                    "--data-path data/{test_data}.csv " \
                    "--load-path save_model_{data}/both/ " \
                    "--log-path test_logs/{data}/both.log --use-pairwise "

    docker_command_list = [docker_command_momentum, docker_command_neither, 
                           docker_command_pairwise, docker_command_both]

    data_index = 6
    GPU_index = 2

    if data_list[data_index] == "0th_ts_train" or data_list[data_index][2:] == "th_ts_train" or data_list[data_index] == "probe":
        for i in range(len(docker_command_list)):
            docker_command_list[i] = docker_command_list[i] + out_c_30
    elif data_list[data_index] == "creditcard":
        for i in range(len(docker_command_list)):
            docker_command_list[i] = docker_command_list[i] + out_c_20
    
    for i in [0, 1, 2, 3]:
        p = Process(target=fun1, args=(docker_command_list[i].format(GPU=GPU_index, test_data=data_list[data_index], data=data_list[data_index]) + criterion_distance,))
        p.start()
        process_list.append(p)
    for i in [0, 1, 2, 3]:
        p = Process(target=fun1, args=(docker_command_list[i].format(GPU=GPU_index, test_data=data_list[data_index], data=data_list[data_index]) + criterion_iforest,))
        p.start()
        process_list.append(p)
    for i in process_list:
        p.join()
    print("测试结束")
