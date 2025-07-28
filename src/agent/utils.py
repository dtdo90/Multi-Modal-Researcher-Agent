import os, io
import wave
from google.genai import Client, types
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv
from PIL import Image
from agent.configuration import Configuration
import re

load_dotenv()

# Initialize client
client = Client(api_key=os.getenv("GEMINI_API_KEY"))
config = Configuration()

def display_gemini_response(response):
    """Extract text from Gemini response and display as markdown with references"""
    console = Console()
    
    # Extract main content
    text = response.candidates[0].content.parts[0].text
    md = Markdown(text)
    console.print(md)
    
    # Get candidate for grounding metadata
    candidate = response.candidates[0]
    
    # Build sources text block
    sources_text = ""
    
    # Display grounding metadata if available
    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
        console.print("\n" + "="*50)
        console.print("[bold blue]References & Sources[/bold blue]")
        console.print("="*50)
        
        # Display and collect source URLs
        if candidate.grounding_metadata.grounding_chunks:
            console.print(f"\n[bold]Sources ({len(candidate.grounding_metadata.grounding_chunks)}):[/bold]")
            sources_list = []
            for i, chunk in enumerate(candidate.grounding_metadata.grounding_chunks, 1):
                if hasattr(chunk, 'web') and chunk.web:
                    title = getattr(chunk.web, 'title', 'No title') or "No title"
                    uri = getattr(chunk.web, 'uri', 'No URI') or "No URI"
                    console.print(f"{i}. {title}")
                    console.print(f"   [dim]{uri}[/dim]")
                    sources_list.append(f"{i}. {title}\n   {uri}")
            
            sources_text = "\n".join(sources_list)
        
        # Display grounding supports (which text is backed by which sources)
        if candidate.grounding_metadata.grounding_supports:
            console.print(f"\n[bold]Text segments with source backing:[/bold]")
            for support in candidate.grounding_metadata.grounding_supports[:5]:  # Show first 5
                if hasattr(support, 'segment') and support.segment:
                    snippet = support.segment.text[:100] + "..." if len(support.segment.text) > 100 else support.segment.text
                    source_nums = [str(i+1) for i in support.grounding_chunk_indices]
                    console.print(f"• \"{snippet}\" [dim](sources: {', '.join(source_nums)})[/dim]")
    
    return text, sources_text


def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Save PCM data to a wave file"""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)

def create_research_report(topic, search_text, video_text, search_sources_text, video_url, configuration=None):
    """Create a comprehensive research report by synthesizing search and video content"""
    
    # Use default values if no configuration provided
    if configuration is None:
        configuration = Configuration()
    
    # Step 1: Create synthesis using Gemini
    synthesis_prompt = f"""
    You are a research analyst. I have gathered information about "{topic}" from two sources:
    
    SEARCH RESULTS:
    {search_text}
    
    VIDEO CONTENT:
    {video_text}
    
    Please create a comprehensive synthesis that:
    1. Identifies key themes and insights from both sources
    2. Highlights any complementary or contrasting perspectives
    3. Provides an overall analysis of the topic based on this multi-modal research
    4. Keep it concise but thorough (3-4 paragraphs)
    
    Focus on creating a coherent narrative that brings together the best insights from both sources.
    """
    
    synthesis_response = client.models.generate_content(
        model=configuration.synthesis_model,
        contents=synthesis_prompt,
        config={
            "temperature": configuration.synthesis_temperature,
        }
    )
    
    synthesis_text = synthesis_response.candidates[0].content.parts[0].text
    
    # Step 2: Create markdown report
    report = f"""# Research Report: {topic}

## Executive Summary

{synthesis_text}

## Video Source
- **URL**: {video_url}

## Additional Sources
{search_sources_text}

---
*Report generated using multi-modal AI research combining web search and video analysis*
"""
    
    return report, synthesis_text


def create_podcast_discussion(topic, search_text, video_text, search_sources_text, video_url, filename, configuration=None):
    """Create a 2-speaker podcast discussion explaining the research topic"""
    
    # Use default values if no configuration provided
    if configuration is None:
        from multi_modal_agent.configuration import Configuration
        configuration = Configuration()
    
    # Step 1: Generate podcast script
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
        
    # # Step 2: Generate TTS audio
    # tts_prompt = f"TTS the following conversation between Mike and Dr. Lisa:\n{podcast_script}"
    
    # response = client.models.generate_content(
    #     model=configuration.tts_model,
    #     contents=tts_prompt,
    #     config=types.GenerateContentConfig(
    #         response_modalities=["AUDIO"],
    #         speech_config=types.SpeechConfig(
    #             multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
    #                 speaker_voice_configs=[
    #                     types.SpeakerVoiceConfig(
    #                         speaker='Mike',
    #                         voice_config=types.VoiceConfig(
    #                             prebuilt_voice_config=types.PrebuiltVoiceConfig(
    #                                 voice_name=configuration.mike_voice,
    #                             )
    #                         )
    #                     ),
    #                     types.SpeakerVoiceConfig(
    #                         speaker='Dr. Lisa',
    #                         voice_config=types.VoiceConfig(
    #                             prebuilt_voice_config=types.PrebuiltVoiceConfig(
    #                                 voice_name=configuration.lisa_voice,
    #                             )
    #                         )
    #                     ),
    #                 ]
    #             )
    #         )
    #     )
    # )
    
    # # Step 3: Save audio file
    # audio_data = response.candidates[0].content.parts[0].inline_data.data
    # wave_file(filename, audio_data, configuration.tts_channel, configuration.tts_rate, configuration.tts_sample_width)
    
    # print(f"Podcast saved as: {filename}")
    # return podcast_script, filename



def parse_transcript_with_sections(transcript_text, sections):
    """Parse a podcast dialogue transcript into segments aligned with provided sections"""
    segments = []
    
    # Split transcript into lines and clean up
    lines = [line.strip() for line in transcript_text.split('\n') if line.strip()]
    
    # Expected speakers for validation
    EXPECTED_SPEAKERS = ["Mike", "Dr. Lisa"]
    
    # Track current section (distribute segments evenly across sections)
    total_dialogue_lines = sum(1 for line in lines if ':' in line and any(speaker in line for speaker in EXPECTED_SPEAKERS))
    segments_per_section = max(1, total_dialogue_lines // len(sections)) if sections else total_dialogue_lines
    current_segment_count = 0
    current_section_idx = 0
    
    for line in lines:
        # Look for speaker dialogue patterns (Speaker: content)
        speaker_match = re.match(r'^([^:]+):\s*(.+)$', line)
        if speaker_match:
            speaker_name = speaker_match.group(1).strip()
            content = speaker_match.group(2).strip()
            
            # Validate and normalize speaker names
            if speaker_name not in EXPECTED_SPEAKERS:
                # Skip invalid speakers or map them if close enough
                if "mike" in speaker_name.lower():
                    speaker_name = "Mike"
                elif "lisa" in speaker_name.lower() or "dr" in speaker_name.lower():
                    speaker_name = "Dr. Lisa"
                else:
                    print(f"⚠️ Skipping unrecognized speaker: {speaker_name}")
                    continue
            
            # Advance to next section periodically to distribute segments
            if sections and current_segment_count >= segments_per_section and current_section_idx < len(sections) - 1:
                current_section_idx += 1
                current_segment_count = 0
            
            # Create segment
            segment = {
                'speaker': speaker_name,
                'content': content,
                'section_idx': current_section_idx,
                'duration': 0.0  # Will be set later when generating audio
            }
            segments.append(segment)
            current_segment_count += 1
    
    print(f"✅ Parsed {len(segments)} segments from transcript across {len(set(seg['section_idx'] for seg in segments))} sections")
    return segments



def generate_image_with_prompt(prompt, save_path):
    """Generate image using Gemini API"""
    # Skip generation if save_path already exists
    if os.path.exists(save_path):
        print(f"⏭️ Skipping generation: {os.path.basename(save_path)}")
        return save_path
        
    try:
        print(f"Attempting AI image generation: {os.path.basename(save_path)}")
        
        response = client.models.generate_content(
            model= config.image_model,
            contents=prompt,  # Use prompt directly as string
            config=types.GenerateContentConfig(response_modalities=['TEXT', 'IMAGE'])
        )
        
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print(part.text)
            elif part.inline_data is not None:
                image = Image.open(io.BytesIO((part.inline_data.data)))
                image.save(save_path)
                return save_path
        
        # If we reach here, no image was generated
        print(f"❌ AI image generation failed: No image produced")
        return None
        
    except Exception as e:
        print(f"❌ AI image generation failed: {e}")
        return None

