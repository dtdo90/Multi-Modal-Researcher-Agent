from typing_extensions import TypedDict
from typing import Optional

class ResearchStateInput(TypedDict):
    """State for the research and podcast generation workflow"""
    topic: str
    video_url: Optional[str]
    output_path: Optional[str]

class ResearchStateOutput(TypedDict):
    """State for the research and podcast generation workflow"""
    report: Optional[str]
    podcast_script: Optional[str]
    podcast_filename: Optional[str]

class ResearchState(TypedDict):
    """State for the research and podcast generation workflow"""
    # Input fields
    topic: str
    video_url: Optional[str]
    output_path: Optional[str]
    speakers: Optional[dict]
    sections: Optional[list]
    speaker_images: Optional[dict]
    section_backgrounds: Optional[dict]
    analysis: Optional[dict]

    # Intermediate results
    search_text: Optional[str]
    search_sources_text: Optional[str]
    video_text: Optional[str]

    # Final outputs
    report: Optional[str]
    synthesis_text: Optional[str]
    podcast_script: Optional[str]
    podcast_filename: Optional[str]

