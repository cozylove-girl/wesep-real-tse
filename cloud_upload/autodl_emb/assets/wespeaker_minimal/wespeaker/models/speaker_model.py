from wespeaker.models import ecapa_tdnn

def get_speaker_model(name):
    if not name.startswith("ECAPA_TDNN"):
        raise ValueError("This EMB runtime supports ECAPA only: " + name)
    return getattr(ecapa_tdnn, name)
