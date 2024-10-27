class ConsoleFormatter:
    """
    Class that represents a formatter that allows to color a text to be displayed in console.

    Attributes:
        colors (dict): Dictionary whose keys are descriptions of the value colors.
    """


    OKGREEN = "\033[92m"
    END = "\033[0m"
    WARNING = "\033[93m"
    OKBLUE = "\033[94m"
    ERROR = "\033[91m"


    def __init__(self):
        self.colors={
        "HEADER":'\033[95m',
        }
        
    def format(self, text, format):
        """
        Given a text and a specified format returns the text with the corresponding color for console.

        Args:
            text (str): Text to be formatted.
            format (str): Format that represents the color to be formatted.

        Raises:
            KeyError: If format is not a key in the dictionary of the attribute colors.

        Returns:
            Returns the text formatted with the color for console corresponding to the format especified.
        """
        return(self.colors[format]+text+self.colors["ENDC"])
    

    @staticmethod
    def okgreen(text: str) -> str:
        return f"{ConsoleFormatter.OKGREEN}{text}{ConsoleFormatter.END}"
    
    
    @staticmethod
    def warning(text: str) -> str:
        return f"{ConsoleFormatter.WARNING}{text}{ConsoleFormatter.END}"
    
    
    @staticmethod
    def okblue(text: str) -> str:
        return f"{ConsoleFormatter.OKBLUE}{text}{ConsoleFormatter.END}"
    
    @staticmethod
    def error(text: str) -> str:
        return f"{ConsoleFormatter.ERROR}{text}{ConsoleFormatter.END}"
