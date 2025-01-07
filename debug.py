import tasks.tasks_sim_v2 as tasks_sim_v2

task = tasks_sim_v2.VeritasQA("en","/mnt/netapp1/Proxecto_NOS/adestramentos/avaliacion/cache")
task.load_data()
print(task)
print(task.dataset[1])

