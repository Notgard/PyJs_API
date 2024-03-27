from pydantic import BaseModel

class ChatbotRequest(BaseModel):
    instruction:        str   | None = None
    iinput:             str   | None = None
    context:            str   | None = None
    instruction_nochat: str
    iinput_nochat:      str   | None = None
    visible_models:     list  | None = None
    max_new_tokens:     int   | None = None
    max_time:           int   | None = None
    repetition_penalty: float | None = None
    do_sample:          bool  | None = None
    temperature:        float | None = None
    h2ogpt_key:         str   | None = None
    prompt_type:        str   | None = None
    num_prompt_tokens:  int   | None = None
    early_stopping:     bool  | None = None
    stream_output:      bool  | None = None
