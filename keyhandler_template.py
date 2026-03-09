import os

# TODO - change this filename to keyhandler.py when you are ready to use it.

class KeyHandler:
    key = ''
    claude_key = ''
    hf_key = ''

    @classmethod
    def set_env_key(cls):

        os.environ['OPENAI_API_KEY'] = cls.key
        os.environ['ANTHROPIC_API_KEY'] = cls.claude_key
        os.environ['HF_API_KEY'] = cls.hf_key
        os.environ['HF_TOKEN'] = cls.hf_key