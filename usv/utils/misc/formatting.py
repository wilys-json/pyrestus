from datetime import datetime

def format_filename(name='', pid='', start_time='', **kwargs)->str:

    elements = [name, pid, start_time]

    remarks = kwargs.get('remarks', '')
    start_time = str(start_time)
    start_time = (start_time if len(start_time) == 0
               else "".join("_".join(start_time.split(" ")).split(":")))

    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    filename = (f'{name}{pid}{start_time}{remarks}'
                if all(elements)
                else timestamp)

    return filename
