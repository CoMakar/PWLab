import json
from typing import Any, List


class JSONConfigError(Exception):
    pass


class InvalidFileContent(JSONConfigError):
    pass


class JSONConfig:
    __read_only: List

    def __setattr__(self, name: str, value: Any):
        if name in self.__read_only:
            raise AttributeError(f"Attribute {self.__class__.__qualname__}.{name} is read-only")
        
        self.__dict__[name] = value
    
    def __init__(self, config: dict, attributes_mode: bool = False):
        self.__dict__[f"_{self.__class__.__name__}__read_only"] = []    
        
        self.__config = config.copy()
        self.__attributes_mode = attributes_mode

        self.__preprocess()    
        self.__schema = self.__generate_schema()
        
        if attributes_mode:
            self.__generate_attributes()

    @property
    def schema(self):
        return self.__schema.copy()
    
    @property
    def attribute_mode(self):
        return self.__attributes_mode
    
    def __preprocess(self):
        for key, val in self.__config.items():
            if isinstance(val, dict):
                self.__config[key] = JSONConfig(val, self.__attributes_mode)
    
    def __generate_schema(self):
        schema = dict.fromkeys(self.__config.keys())
        
        for key, val in self.__config.items():
            if isinstance(val, JSONConfig):
                schema[key] = val.schema
            else:
                schema[key] = type(val)
    
        return schema
    
    def __generate_attributes(self):
        for key, val in self.__config.items():
            key = str(key)
            
            if not key.isidentifier():
                raise SyntaxError(f"'{key}' is not a valid python identifier")
            
            if hasattr(self, key):
                raise AttributeError(f"Attribute {self.__class__.__qualname__}.{key} already exists")
            
            setattr(self, key, val)

            self.__read_only.append(key)
        
    def get(self, key: str, default=None):
        """
        Returns:
            value or default
        """        
        return self.__config.get(key, default)
          
    @classmethod
    def load(cls, file_path: str, attributes_mode: bool = False) -> 'JSONConfig':
        """
        Returns:
            JSONConfig with content from file, if .json file is invalid,
            empty JSONConfig will be returned
        """
        try:
            with open(file_path, 'r', encoding='utf8') as config_file:
                json_data = json.load(config_file)
        except json.JSONDecodeError:
            json_data = {}
            
        return JSONConfig(json_data, attributes_mode)
    
    def save(self, file_path: str):
        with open(file_path, 'w', encoding='utf-8') as config_file:
            config_file.write(json.dumps(self.as_dict(), indent=4)) 
            
    def as_dict(self):
        dictionary = self.__config.copy()
        sections = filter(lambda e: isinstance(e[1], JSONConfig), dictionary.items())
    
        for key, val in sections:
            dictionary[key] = val.as_dict()
            
        return dictionary
        
    def __getitem__(self, key):
        return self.__config[key]
            
    def __str__(self):
        return str(self.as_dict())
    
    def __repr__(self):
        return f"{self.__class__.__qualname__}({self.as_dict()}, {self.__attributes_mode})"
