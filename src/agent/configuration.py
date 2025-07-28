"""Configuration settings for the research and podcast generation app"""
import os
from dataclasses import dataclass, fields
from typing import Optional, Any
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig

@dataclass(kw_only=True)
class Configuration:
    """LangGraph configuration for the deep research agent."""
    # model settings
    search_model: str="gemini-2.5-flash"     # web search model (cheaper, faster)
    synthesis_model: str="gemini-2.5-flash" # synthesis model (better for analysis)
    video_model: str="gemini-2.5-flash"     # video analysis model (better for multimodal)
    tts_model: str="gemini-2.5-flash-preview-tts"
    image_model: str="gemini-2.0-flash-preview-image-generation"

    # temperature settings
    search_temperature: float=0.0 # factual search
    synthesis_temperature: float=0.3 # balanced synthesis
    podcast_temperature: float=0.4 # creative dialogue

    # TTS configuration
    mike_voice: str= "Puck"
    lisa_voice: str= "Kore"
    tts_channel: int=1 # 0 for mono, 1 for stereo
    tts_rate: int =24000 # sample rate in Hz
    tts_sample_width: int =2 # sample width in bytes

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig]) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable=config["configurable"] if config and "configurable" in config else {}
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls) if f.init
        }
        return cls(**{k: v for k,v in values.items() if v})