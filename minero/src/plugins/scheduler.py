from apscheduler.schedulers.background import BackgroundScheduler


# Espera recibir una función y un intervalo de tiempo en segundos que define cada cuanto tiempo la invocará
def start_cronjob(task, interval):
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=task, trigger="interval", seconds=interval)
    scheduler.start()
