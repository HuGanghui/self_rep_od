import os
from multiprocessing import Process


def fun1(command):
    os.system(command)


if __name__ == '__main__':
    # 多进程利用多个GPU训练模型
    process_list = []
    data_list = ["apascal", "bank-additional-full_normalised", "lung-1vs5", "probe", "secom"]

    docker_command_neither = "CUDA_VISIBLE_DEVICES={GPU} python train.py " \
                     "--data-path data/{data}.csv " \
                     "--save-path save_model_{data}/neither/ " \
                     "--log-path logs/{data}/neither.log"  

    docker_command_pairwise = "CUDA_VISIBLE_DEVICES={GPU} python train.py " \
                     "--data-path data/{data}.csv " \
                     "--save-path save_model_{data}/pairwise/ " \
                     "--log-path logs/{data}/pairwise.log --use-pairwise" 

    docker_command_momentum = "CUDA_VISIBLE_DEVICES={GPU} python train.py " \
                    "--data-path data/{data}.csv " \
                    "--save-path save_model_{data}/momentum/ " \
                    "--log-path logs/{data}/momentum.log --use-momentum" 

    docker_command_both = "CUDA_VISIBLE_DEVICES={GPU} python train.py " \
                    "--data-path data/{data}.csv " \
                    "--save-path save_model_{data}/both/ " \
                    "--log-path logs/{data}/both.log --use-pairwise --use-momentum" 

    docker_command_list = [docker_command_neither, docker_command_pairwise, 
                            docker_command_momentum, docker_command_both]                                 
    # for i in range(4):
    #     p = Process(target=fun1, args=(docker_command_list[i].format(GPU=i, data=data_list[2]),))
    #     p.start()
    #     process_list.append(p)
    for i in range(4):
        p = Process(target=fun1, args=(docker_command_list[i].format(GPU=4 + i, data=data_list[3]),))
        p.start()
        process_list.append(p)
    for i in process_list:
        p.join()
    print("测试结束")
