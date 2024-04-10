from typing import Dict, Union
import yaml


class RoleConfigLoader:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.role_config_map = self.load_config()
        self.t2s_weights_path = self.role_config_map.get("t2s_weights_path", None)
        self.vits_weights_path = self.role_config_map.get("vits_weights_path", None)
        self.ref_wav = self.role_config_map.get("ref_wav", None)
        self.ref_text = self.role_config_map.get("ref_text", None)

    def load_config(self) -> Dict[str, Dict[str, str]]:
        with open(self.config_file, 'r', encoding='utf-8') as f:
            role_config_map = yaml.safe_load(f)
        return role_config_map

    def get_configs_for_role(self, role: str) -> Union[Dict[str, str], None]:
        return self.role_config_map.get(role, None)