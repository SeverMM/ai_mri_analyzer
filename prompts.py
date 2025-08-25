"""Prompt templates for ai_mri_analyzer."""

SYSTEM_PROMPT = (
    "You are a board-certified radiologist with extensive experience in MRI interpretation. "
    "Carefully analyze the provided MRI slices and follow the user's task instructions."
    "Note: Metadata text visible in the images may be in Romanian - interpret it accordingly."
)

USER_TEMPLATE = (
    "Analyse the following MRI series:\n\n"
    "• Series ID: {series_id}\n"
    "• Sequence type: {sequence_type}\n"
    "• Slice count sent: {slice_count}\n"
    "• Patient context: {patient_context}\n"
    "• Clinical question: Confirm or refute possible {previous_ai_flag} noted in preliminary AI review.\n\n"
    "TASK  \n"
    "Return a JSON object with this exact structure:\n\n"
    "{json_schema}\n\n"
    "If no abnormal findings, return \"findings\": [] but still fill impression & recommendations."
) 