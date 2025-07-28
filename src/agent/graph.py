"""LangGraph implementation of the research and podcast generation workflow
Install dependencies and start the LangGraph server:
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
Example run: 
Topic: Overview on the current state of AGI
Video url: https://www.youtube.com/watch?v=4__gg83s_Do 
"""


import os, json, re
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from google.genai import types
from pathlib import Path

from agent.state import ResearchState, ResearchStateInput, ResearchStateOutput
from agent.utils import display_gemini_response, create_podcast_discussion, create_research_report, client, parse_transcript_with_sections, generate_image_with_prompt, config
from agent.audios import generate_audio_and_update_segments, assign_images_to_segments, create_video
from agent.configuration import Configuration

from langsmith import traceable

@traceable(run_type="llm", name="Web Search")
def search_research_node(state: ResearchState, config: RunnableConfig)-> dict:
    """Node that performs web search research on a topic"""
    configuration=Configuration.from_runnable_config(config) # does this allow config to be adjustable on langsmith?
    topic=state["topic"]

    search_response=client.models.generate_content(
        model=configuration.search_model,
        contents=f"Research this topic and give me an overview: {topic}",
        config={"tools": [{"google_search":{}}],
                "temperature": configuration.search_temperature
        },
    )
    search_text, search_sources_text=display_gemini_response(search_response)

    return {
        "search_text": search_text,
        "search_sources_text": search_sources_text
    }

@traceable(run_type="llm", name="Youtube Video Analysis")
def analyze_video_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that analyzes video content if URL is provided"""
    configuration=Configuration.from_runnable_config(config)
    video_url=state.get("video_url")
    topic=state["topic"]

    if not video_url:
        return {"video_text": "No video provided for analysis."}
    
    video_response=client.models.generate_content(
        model=configuration.video_model,
        contents=types.Content(
            parts=[
                types.Part(file_data=types.FileData(file_uri=video_url)),
                types.Part(text=f"Based on the video content, give me an overview of this topic:\n{topic}")
            ]
        )
    )
    video_text, _= display_gemini_response(video_response) # _ = citations
    return {"video_text": video_text}

@traceable(run_type="llm", name="Create Podcast Script")
def create_podcast_transcript(state: ResearchState, config: RunnableConfig) -> str:
    """Create a 2-speaker podcast discussion explaining the research topic"""
    configuration = Configuration.from_runnable_config(config)

    search_text= state.get("search_text", "")
    video_text= state.get("video_text", "")
    topic= state.get("topic", "")

    # Generate podcast script
    script_prompt = f"""
    Create a natural, engaging podcast conversation between Dr. Lisa (research expert) and Mike (curious interviewer) about "{topic}".
    
    Use this research content:
    
    SEARCH FINDINGS:
    {search_text}
    
    VIDEO INSIGHTS:
    {video_text}
    
    Format as a dialogue with:
    - Mike introducing the topic and asking questions
    - Dr. Lisa explaining key concepts and insights
    - Natural back-and-forth discussion (5-7 exchanges)
    - Mike asking follow-up questions
    - Dr. Lisa synthesizing the main takeaways
    - Keep it conversational and accessible (3-4 minutes when spoken)
    
    Format exactly like this:
    Mike: [opening question]
    Dr. Lisa: [expert response]
    Mike: [follow-up]
    Dr. Lisa: [explanation]
    [continue...]
    """
    
    script_response = client.models.generate_content(
        model=configuration.synthesis_model,
        contents=script_prompt,
        config={"temperature": configuration.podcast_temperature}
    )
    
    podcast_script = script_response.candidates[0].content.parts[0].text
    # save the script to a file
    podcast_dir = os.path.join(os.path.dirname(__file__), "..", "..", "podcast")
    os.makedirs(podcast_dir, exist_ok=True)
    script_path = os.path.join(podcast_dir, "script.txt")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(podcast_script)

    return {"podcast_script": podcast_script}

    



@traceable(run_type="llm", name="Segment Transcript")
def segment_transcript(state: ResearchState, config: RunnableConfig) -> dict:
    """Use LLM to intelligently analyze and segment the transcript"""
    configuration = Configuration.from_runnable_config(config)
    transcript_text=state.get("podcast_script", "")

    analysis_prompt = f"""
    Analyze this podcast transcript and provide:
    1. A list of distinct speakers and their characteristics/roles
    2. Intelligent segmentation of the content into thematic sections
    3. Key topics and themes discussed in each section
    
    Transcript:
    {transcript_text}
    
    Please return a JSON response with this structure:
    {{
        "speakers": {{
            "speaker_name": {{
                "role": "description of their role/expertise",
                "characteristics": "physical and personality traits for image generation"
            }}
        }},
        "sections": [
            {{
                "title": "section title",
                "start_text": "first few words to identify start",
                "end_text": "last few words to identify end", 
                "theme": "main theme/topic",
                "mood": "visual mood/atmosphere",
                "key_concepts": ["concept1", "concept2"],
                "duration_estimate": estimated_seconds
            }}
        ]
    }}
    """
    
    response = client.models.generate_content(
        model=configuration.synthesis_model,
        contents=analysis_prompt,
        config={"temperature": configuration.synthesis_temperature}
    )
    
    # Extract JSON from response 
    response_text = response.candidates[0].content.parts[0].text
            
    # Try to extract JSON using regex first
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            analysis = json.loads(json_match.group())
            print("✅ Successfully parsed JSON using regex extraction")
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse extracted JSON: {e}")
            raise
    else: # If no JSON found with regex, parse the entire response            
        analysis = json.loads(response_text)
        print("✅ Successfully parsed JSON from full response")
        
    return {"analysis": analysis}



# ========= Generate Images =========
# Get the project root directory (assuming we're in src/agent/)
project_root = Path(__file__).parent.parent.parent
speakers_dir = project_root / "podcast" / "images" / "speakers"
backgrounds_dir = project_root / "podcast" / "images" / "backgrounds"
# Create directories if they don't exist
os.makedirs(speakers_dir, exist_ok=True)
os.makedirs(backgrounds_dir, exist_ok=True)

@traceable(run_type="llm", name="Generate Speaker Images")
def generate_speaker_images(state: ResearchState, config: RunnableConfig):
    """Generate images for each speaker using LLM-generated prompts"""
    # Convert incoming config (dict) to Configuration for attribute access
    configuration = Configuration.from_runnable_config(config)

    # Accept speakers directly on state or nested under the analysis key
    analysis = state.get("analysis", {})
    speakers_info = analysis.get("speakers", {}) 
    speaker_images = {}
    
    for speaker_name, info in speakers_info.items():
        print(f"Generating image for speaker: {speaker_name}")

        # Check if image already exists 
        safe_name = "".join(c for c in speaker_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        save_path = os.path.join(speakers_dir, f"{safe_name}.png")
        if os.path.exists(save_path):
            print(f"✅ Image already exists, skipping generation: {os.path.basename(save_path)}")
            speaker_images[speaker_name] = save_path
            continue

        # Generate image prompt using LLM
        prompt_request = f"""
        Create a detailed image generation prompt for a professional podcast speaker with these characteristics:
        Name: {speaker_name}
        Role: {info['role']}
        Characteristics: {info['characteristics']}
        
        Focus on: professional appearance, clear facial features, appropriate background, good lighting.
        Keep it realistic and professional.
        
        Return only the image generation prompt, no additional text.
        """
        
        prompt_response = client.models.generate_content(
            model= configuration.synthesis_model,
            contents=prompt_request,
            config={"temperature": configuration.synthesis_temperature}
        )
        
        image_prompt = prompt_response.candidates[0].content.parts[0].text.strip()
        print(f"Generated prompt for {speaker_name}:\n{image_prompt}")
        
        # Generate the actual speaker image
        try:
            speaker_image = generate_image_with_prompt(
                image_prompt,
                save_path
            )
            if speaker_image:
                speaker_images[speaker_name] = speaker_image
                print(f"✅ Successfully generated image for {speaker_name}: {speaker_image}")
            else:
                print(f"❌ Failed to generate image for {speaker_name}")
        except Exception as e:
            print(f"❌ Error generating image for {speaker_name}: {e}")
    
    return {
        "speaker_images": speaker_images,
        "speakers": speakers_info,
    }

@traceable(run_type="llm", name="Generate Background Images")
def generate_section_backgrounds(state: ResearchState, config: RunnableConfig):
    """Generate background images for each section using LLM analysis"""
    # Convert config dict into Configuration object
    configuration = Configuration.from_runnable_config(config)

    # Accept sections directly on state or nested under analysis
    analysis=state.get("analysis", {})
    sections = analysis.get("sections", [])
    section_backgrounds: dict[str, str] = {}
    
    for i, section in enumerate(sections):
        print(f"Generating background for section {i}: {section.get('title', 'Unknown')}")
        # Generate background image prompt using LLM
        background_prompt_response = f"""
        Create a detailed image generation prompt for a podcast video background based on this section:
        
        Title: {section['title']}
        Theme: {section['theme']}
        Mood: {section['mood']}
        Key Concepts: {', '.join(section['key_concepts'])}
        
        The image should be:
        - Abstract and not distracting from speakers
        - Professional and modern
        - Relevant to the theme and concepts
        - Suitable as a video background (16:9 aspect ratio)
        - Visually appealing but not overwhelming
        
        Return only the image generation prompt, no additional text.
        """
        
        prompt_response = client.models.generate_content(
            model= configuration.synthesis_model,
            contents=background_prompt_response,
            config={"temperature": configuration.synthesis_temperature}
        )
        
        background_prompt = prompt_response.candidates[0].content.parts[0].text.strip()
        print(f"Generated background prompt {i}:\n{background_prompt}")
        
        # Generate the actual background image
        try:
            # Create meaningful filename
            safe_title = "".join(c for c in section['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            save_path = os.path.join(backgrounds_dir, f"section_{i:02d}_{safe_title}.png")
            
            background_image = generate_image_with_prompt(
                background_prompt,
                save_path
            )
            if background_image:
                section_backgrounds[f"section_{i:02d}"] = background_image
                print(f"✅ Successfully generated background for section {i}: {background_image}")
            else:
                print(f"❌ Failed to generate background for section {i}")
        except Exception as e:
            print(f"❌ Error generating background for section {i}: {e}")
    
    # Persist backgrounds so downstream nodes see them via `state.get("section_backgrounds")`
    return {
        "section_backgrounds": section_backgrounds,
        "sections": sections,
    }


@traceable(run_type="llm", name="Create Podcast")
def create_video_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Main method to generate complete video from transcript using LLM-driven approach"""
    configuration = Configuration.from_runnable_config(config)
    audio_file = None
    transcript_text = state.get("podcast_script", "")

    # Create absolute path for output
    if state.get("output_path"):
        output_path = state.get("output_path")
    else:
        # Use absolute path based on project root
        project_root = Path(__file__).parent.parent.parent
        podcast_dir = project_root / "podcast"
        podcast_dir.mkdir(parents=True, exist_ok=True)
        output_path = podcast_dir / "podcast_video.mp4"
    
    try:
        print("Analyzing transcript with LLM...")
        # Analyze transcript with LLM to get sections and speakers
        analysis = state.get("analysis", {})
        sections = analysis.get("sections", [])
        speaker_images = state.get("speaker_images", {})
        section_backgrounds = state.get("section_backgrounds", {})

        print("Parsing transcript into segments...")

        # Parse transcript into segments aligned with sections
        segments = parse_transcript_with_sections(transcript_text, sections)
        
        print("Assigning images to segments...")
        # Assign images to segments
        segments = assign_images_to_segments(segments, section_backgrounds)
        
        print("Generating TTS audio with accurate segment durations...")
        audio_file, segments = generate_audio_and_update_segments(segments)
        
        print("Creating final video...")
        final_video_path = create_video(segments, speaker_images, output_path, audio_file)
        
        print(f"Video created successfully: {final_video_path}")
        return {
            "podcast_filename": final_video_path
        }
        
    except Exception as e:
        print(f"❌ Error generating video: {e}")
        raise
    finally:
        # Clean up temporary audio file only if we generated it ourselves
        if audio_file and os.path.exists(audio_file) and (audio_file.startswith('/tmp') or audio_file.startswith('/var/folders')):
            try:
                os.unlink(audio_file)
                print(f"Cleaned up temporary audio file: {audio_file}")
            except Exception as e:
                print(f"⚠️ Could not clean up temporary file {audio_file}: {e}")



def should_analyze_video(state: ResearchState) -> str:
    """Conditional edge to determine if video analysis should be performed"""
    if state.get("video_url"):
        return "analyze_video" # go to analyze_video node
    else:
        return "create_report" # go to create_report node
    
def build_graph() -> StateGraph:
    """Create the research workflow graph"""
    # Initialize the graph with configuration schema
    graph = StateGraph(
        ResearchState,
        input_schema=ResearchStateInput,
        output_schema=ResearchStateOutput,
        config_schema=Configuration
    )
    # Add nodes
    graph.add_node("search_research", search_research_node)
    graph.add_node("analyze_video", analyze_video_node)
    graph.add_node("create_podcast_transcript", create_podcast_transcript)
    graph.add_node("segment_transcript", segment_transcript)
    graph.add_node("generate_speaker_images", generate_speaker_images)
    graph.add_node("generate_section_backgrounds", generate_section_backgrounds)
    graph.add_node("create_video", create_video_node)

    # Add edges
    graph.add_edge(START, "search_research")
    graph.add_conditional_edges(
        "search_research",
        should_analyze_video,
        {
            "analyze_video": "analyze_video",
            "create_podcast_transcript": "create_podcast_transcript"
        }
    )
    graph.add_edge("analyze_video", "create_podcast_transcript")
    graph.add_edge("create_podcast_transcript", "segment_transcript")
    graph.add_edge("segment_transcript", "generate_speaker_images")
    graph.add_edge("segment_transcript", "generate_section_backgrounds")
    graph.add_edge("generate_speaker_images", "create_video")
    graph.add_edge("generate_section_backgrounds", "create_video")
    graph.add_edge("create_video", END)    
    
    return graph

def create_graph():
    """Create and compile the research graph"""
    graph = build_graph()
    graph = graph.compile()
    # save png file for graph
    graph_png=graph.get_graph().draw_mermaid_png()
    
    # Create podcast directory if it doesn't exist
    podcast_dir = Path(__file__).parent.parent.parent / "podcast"
    podcast_dir.mkdir(parents=True, exist_ok=True)    
    graph_path = podcast_dir / "graph.png"

    with open(graph_path, "wb") as f:
        f.write(graph_png)
    return graph

if __name__ == "__main__":
    topic="Overview on the current state of AGI"
    video_url="https://www.youtube.com/watch?v=4__gg83s_Do"
    input_state=ResearchStateInput(topic=topic, video_url=video_url)
    graph=create_graph()
    for event in graph.stream(input_state, stream_mode="values"):
        # LangGraph returns a dict with keys depending on stream_mode
        if isinstance(event, dict):
            state = event.get("value", event)  # fallback to whole dict
        elif isinstance(event, (list, tuple)):
            state = event[-1]
        else:
            state = event

        if hasattr(state, "pretty_print"):
            state.pretty_print()
        else:
            print(state)