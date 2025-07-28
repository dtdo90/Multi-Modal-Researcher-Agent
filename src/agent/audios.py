from moviepy import AudioFileClip, CompositeVideoClip, ImageClip
import tempfile
import os
from dotenv import load_dotenv
from agent.utils import client, config


load_dotenv()

def generate_audio_and_update_segments(segments):
    """Generate TTS audio for each segment, measure actual durations, and concatenate"""
    import wave
    from google.genai import types
    
    print(f"Generating TTS audio for {len(segments)} segments...")
    segment_audio_files = []
    updated_segments = []
    
    for i, segment in enumerate(segments):
        print(f"Generating audio for segment {i+1}/{len(segments)}: {segment['speaker']}")
        
        # Generate TTS audio for this segment
        tts_prompt = f"{segment['speaker']}: {segment['content']}"
        
        response = client.models.generate_content(
            model= config.tts_model,
            contents=tts_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=config.mike_voice if segment['speaker'].lower() == 'mike' else config.lisa_voice,
                        )
                    )
                )
            )
        )
        
        # Save segment audio to temporary file
        segment_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio_data = response.candidates[0].content.parts[0].inline_data.data
        
        # Write audio data to file using wave module
        with wave.open(segment_audio_file.name, 'wb') as wav_file:
            wav_file.setnchannels(config.tts_channel if config.tts_channel > 0 else 1)
            wav_file.setsampwidth(config.tts_sample_width)
            wav_file.setframerate(config.tts_rate)
            wav_file.writeframes(audio_data)
        
        # Measure actual duration
        with wave.open(segment_audio_file.name, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            actual_duration = frames / float(rate)
        
        # Update segment with actual duration
        segment_copy = segment.copy()
        segment_copy['duration'] = actual_duration
        updated_segments.append(segment_copy)
        segment_audio_files.append(segment_audio_file.name)
        
        print(f"Generated {actual_duration:.1f}s audio for {segment['speaker']}\n")
    
    print("Concatenating all audio segments...")
    # Concatenate all segment audio files
    final_audio_file = concatenate_audio_files(segment_audio_files)
    
    # Clean up temporary segment files
    for audio_file in segment_audio_files:
        os.unlink(audio_file)
        
    total_duration = sum(seg['duration'] for seg in updated_segments)
    print(f"✅ Generated complete audio file ({total_duration:.1f} seconds)")
    
    return final_audio_file, updated_segments

def concatenate_audio_files(audio_files):
    """Concatenate multiple WAV files into one"""
    import wave
    
    if not audio_files:
        return None
    
    final_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    
    # Open the first file to get audio parameters
    with wave.open(audio_files[0], 'rb') as first_wave:
        params = first_wave.getparams()
        
    # Create output file with same parameters
    with wave.open(final_audio_file.name, 'wb') as output_wave:
        output_wave.setparams(params)
        
        # Concatenate all files
        for audio_file in audio_files:
            with wave.open(audio_file, 'rb') as input_wave:
                output_wave.writeframes(input_wave.readframes(input_wave.getnframes()))
    
    return final_audio_file.name

def assign_images_to_segments(segments, section_backgrounds):
    """Assign appropriate background images to each segment based on their section"""
    
    for segment in segments:
        section_idx = segment.get('section_idx', 0)
        section_key = f"section_{section_idx:02d}"
                
        if section_key in section_backgrounds and section_backgrounds[section_key]:
            segment['background'] = section_backgrounds[section_key]
            print(f"✅ Assigned background to segment: {section_backgrounds[section_key]}")
        else:
            # Fallback to first available background
            available_backgrounds = [bg for bg in section_backgrounds.values() if bg]
            if available_backgrounds:
                segment['background'] = available_backgrounds[0]
                print(f"⚠️ Using fallback background: {available_backgrounds[0]}")
            else:
                # No backgrounds available - just log and set to None
                print(f"❌ No background images available for segment '{segment['speaker']}' in section {section_idx}")
                segment['background'] = None
    
    return segments

    
def create_video(segments, speaker_images, output_path, audio_file=None):
    """Combine audio, images, and speaker images into final video"""
    video_clips = []
    current_time = 0
    
    print(f"Creating video with {len(segments)} segments...")
    
    for i, segment in enumerate(segments):
        print(f"Processing segment {i+1}/{len(segments)}: {segment['speaker']} at {current_time:.1f}s")
        print(f"🔍 DEBUG: Segment background: {segment.get('background')}")
        
        # Create background clip only if background image exists
        if segment.get('background'):
            bg_clip = ImageClip(segment['background'], duration=segment['duration'])
            bg_clip = bg_clip.with_start(current_time)
            video_clips.append(bg_clip)
            print(f"✅ Added background clip for segment {i+1}")
        else:
            print(f"⚠️ Skipping background for segment {i+1} - no image available")
        
        # Add speaker image if available
        speaker_name = segment['speaker']
        print(f"🔍 DEBUG: Looking for speaker '{speaker_name}' in speaker_images")
        if speaker_name in speaker_images and speaker_images[speaker_name]:
            speaker_clip = ImageClip(speaker_images[speaker_name], duration=segment['duration'])
            # Standardize speaker image size - same width and height for consistency
            speaker_clip = speaker_clip.resized((200, 200))  # Fixed size for all speakers
            # Position speaker image in the center of the screen
            # speaker_clip = speaker_clip.with_start(current_time).with_position("center")
            speaker_clip = speaker_clip.with_start(current_time).with_position(("center", 0))
            video_clips.append(speaker_clip)
            print(f"✅ Added speaker {speaker_name} from {current_time:.1f}s to {current_time + segment['duration']:.1f}s")
        else:
            print(f"⚠️ No speaker image available for {speaker_name}")
        
        current_time += segment['duration']
    
    if not video_clips:
        raise ValueError("❌ No video clips were created - cannot generate video")
    
    print("Combining all video clips...")
    # Combine all video clips
    final_video = CompositeVideoClip(video_clips)
    
    # Add audio if available
    if audio_file and os.path.exists(audio_file):
        print(f"Adding audio from: {audio_file}")
        audio_clip = AudioFileClip(audio_file)
        # Ensure audio duration matches video duration
        if audio_clip.duration > final_video.duration:
            audio_clip = audio_clip.subclipped(0, final_video.duration)  # Fixed MoviePy v2 syntax
        elif audio_clip.duration < final_video.duration:
            print(f"⚠️ Audio ({audio_clip.duration:.1f}s) is shorter than video ({final_video.duration:.1f}s)")
        
        final_video = final_video.with_audio(audio_clip)
        print("✅ Audio successfully added to video")
    else:
        print("⚠️ No audio file provided or file doesn't exist - generating silent video")
    
    # Write final video
    print(f"Writing video to {output_path}...")
    final_video.write_videofile(
        output_path,
        fps=24,
        codec='libx264',
        audio_codec='aac' if audio_file and os.path.exists(audio_file) else None
    )
    
    print("Video generation completed!")
    
    # Clean up audio clip if used
    if 'audio_clip' in locals():
        audio_clip.close()
    
    return output_path
