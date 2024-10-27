from .qrcode_detection.QrCodeScanner import QrCodeScanner
from ..config import VisionConfiguration

def initialize(camera: str, config: VisionConfiguration):
    QrCodeScanner(camera)
