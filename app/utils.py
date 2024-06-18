import asyncio
import ast

async def h2o_stream_output(job):
    text_old = ''
    while not job.done():
        outputs_list = job.communicator.job.outputs
        if outputs_list:
            res = job.communicator.job.outputs[-1]
            res_dict = ast.literal_eval(res)
            text = res_dict['response']
            new_text = text[len(text_old):]
            if new_text:
                yield new_text
                text_old = text
            await asyncio.sleep(0.5)
    # handle case if never got streaming response and already done
    res_final = job.outputs()
    if len(res_final) > 0:
        res = res_final[-1]
        res_dict = ast.literal_eval(res)
        text = res_dict['response']
        new_text = text[len(text_old):]
        yield new_text
