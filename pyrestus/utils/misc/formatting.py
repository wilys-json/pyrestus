from datetime import datetime

def format_filename(name='', pid='', start_time='', **kwargs)->str:

    remarks = kwargs.get('remarks', '')
    start_time = str(start_time)
    start_time = (start_time if len(start_time) == 0
               else "_".join(start_time.split(" ")))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    filename = (f'{name}{pid}{start_time}{remarks}'
                if all([name, pid, start_time])
                else timestamp)

    return filename
