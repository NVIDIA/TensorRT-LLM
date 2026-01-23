from tensorrt_llm import logger


def get_local_ip() -> str:
    try:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
            if not ip.startswith("127."):
                return ip

    except (ImportError, OSError, ValueError):
        logger.error("Failed to get local IP via UDP socket method.")
        pass

    try:
        import netifaces

        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface)
            if netifaces.AF_INET in addrs:
                for addr in addrs[netifaces.AF_INET]:
                    ip = addr.get("addr", "")
                    if not ip.startswith("127.") and not ip.startswith("169.254"):
                        return ip
    except (ImportError, OSError, ValueError) as e:
        logger.error(f"Failed to get local IP via netifaces: {e}")
        pass

    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if not ip.startswith("127."):
            return ip
    except (OSError, ValueError):
        logger.error("Failed to get local IP via hostname resolution.")
        pass

    return "127.0.0.1"
