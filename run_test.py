import os

def fun1(command):
    os.system(command)


if __name__ == '__main__':
    # 使用对应数据集的对应模型来测试不同的方法
    process_list = []
    data_list = ["0th_ts_train", "apascal", "bank-additional-full_normalised", "probe"]

    criterion_distance = "--criterion distance"

    criterion_lof = "--criterion lof"

    criterion_iforest = "--criterion iforest"

    out_c_30 = "--out-c 30 "

    docker_command_neither = "CUDA_VISIBLE_DEVICES={GPU} python test.py " \
                     "--data-path data/{data}.csv " \
                     "--load-path save_model_{data}/neither/ " \
                     "--log-path test_logs/{data}/neither.log "  

    docker_command_pairwise = "CUDA_VISIBLE_DEVICES={GPU} python test.py " \
                     "--data-path data/{data}.csv " \
                     "--load-path save_model_{data}/pairwise/ " \
                     "--log-path test_logs/{data}/pairwise.log " 

    docker_command_momentum = "CUDA_VISIBLE_DEVICES={GPU} python test.py " \
                    "--data-path data/{data}.csv " \
                    "--load-path save_model_{data}/momentum/ " \
                    "--log-path test_logs/{data}/momentum.log " 

    docker_command_both = "CUDA_VISIBLE_DEVICES={GPU} python test.py " \
                    "--data-path data/{data}.csv " \
                    "--load-path save_model_{data}/both/ " \
                    "--log-path test_logs/{data}/both.log "

    gpu_index = 6
    data_index = 0

    if data_list[data_index] == "0th_ts_train" or "probe":
        docker_command_both += out_c_30
        docker_command_pairwise += out_c_30
        docker_command_momentum += out_c_30
        docker_command_neither += out_c_30

    fun1(docker_command_neither.format(GPU=gpu_index, data=data_list[data_index]) + criterion_distance)
    fun1(docker_command_neither.format(GPU=gpu_index, data=data_list[data_index]) + criterion_iforest)

    fun1(docker_command_pairwise.format(GPU=gpu_index, data=data_list[data_index]) + criterion_distance)
    fun1(docker_command_pairwise.format(GPU=gpu_index, data=data_list[data_index]) + criterion_iforest)
    
    fun1(docker_command_momentum.format(GPU=gpu_index, data=data_list[data_index]) + criterion_distance)
    fun1(docker_command_momentum.format(GPU=gpu_index, data=data_list[data_index]) + criterion_iforest)

    fun1(docker_command_both.format(GPU=gpu_index, data=data_list[data_index]) + criterion_distance)
    fun1(docker_command_both.format(GPU=gpu_index, data=data_list[data_index]) + criterion_iforest)