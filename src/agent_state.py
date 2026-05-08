from typing import TypedDict, Annotated
import operator

from typing import TypedDict


class AgentState(TypedDict):
    question:        str
    plan:            str
    context:         list
    answer:          str
    sources:         list
    approved:        bool
    critic_feedback: str
    attempts:        int
    chat_history:    list