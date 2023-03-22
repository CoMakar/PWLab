import json

class JSONConfig:
    class FileCorruptedError(Exception):
        ...
    
    def __init__(self, default_config: dict = {}):
        self.config = default_config
        self.schema = self._get_schema(default_config)
        
                
    def _get_schema(self, base_dict: dict, recursion_level: int = 4):
        schema = dict.fromkeys(base_dict.keys())
        
        if recursion_level != 0:
            for key in schema:
                elem = base_dict[key]
                schema[key] = type(elem)
                if isinstance(elem, dict):
                    schema[key] = self._get_schema(elem, recursion_level-1)
                    
        return schema
    
    
    def _get_exc_none(exc: Exception):
        def layer(func):
            def wrapper(self, *args):
                try:
                    return func(self, *args)
                except exc:
                    return None
                
            return wrapper
        
        return layer
       
        
    def get(self, key: str):
        """
        @returns: the value as is or raises KeyError
        """
        return self.config[key]
    
    
    def get_or_none(self, key: str):
        """
        @returns: value as is or None
        """        
        return self.config.get(key, None)
    
    
    def get_int(self, key: str):
        """
        @returns: value as int
        """
        return int(self.get(key))
    
    
    def get_bool(self, key: str):
        """
        @returns: value as bool
        """
        return bool(self.get(key))
    
    
    def get_float(self, key: str):
        """
        @returns: value as float
        """
        return float(self.get(key))
    
    
    def section(self, section_name: str):
        if isinstance(sect := self.get(section_name), dict):
            return JSONConfig(sect)
        else: 
            raise ValueError("Not a section")
        
            
    @classmethod
    def load(cls, file_path: str) -> 'JSONConfig':
        try:
            with open(file_path, 'r', encoding='utf8') as config_file:
                json_data = json.load(config_file)
        except json.JSONDecodeError:
            raise JSONConfig.FileCorruptedError
            
        return JSONConfig(json_data)
    
    def save(self, file_path: str):
        with open(file_path, 'w', encoding='utf-8') as config_file:
            config_file.write(json.dumps(self.config, indent=4)) 
            
    
    def __str__ (self) -> str:
        return str(self.config)