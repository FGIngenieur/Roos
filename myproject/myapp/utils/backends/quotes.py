import os
cwd = os.getcwd()
os.chdir('../libs/')
from quotesLibs import *
os.chdir(cwd)
from quotationEngine import *

class QuoteProject:
    """
    A template class for handling data, performing actions,
    and encapsulating related logic.
    """
    _PUBLIC_ID_REGEX = re.compile(r"^[0-9a-fA-F]{7}[0-9a-fA-F]{2}$")

    def __init__(
                self,
                project_name : str,
                project_type : Optional[Literal["estimate", "quote"]],
                _project_id : int,
                _project_data : pd.DataFrame
                ) -> None:
        """
        Initialize the class.

        Args:
            project_name (str): String referring to the project's name.
            project (str, optional): String that should be either "estimate" or "quote", defaults to "quote".
        """
        print(f"Initialization of the new project >>>")
        self._project_name = project_name
        
        if project_type is None:
            project_type = "quote"

        self._project_type = project_type
        self._project_id = __generate_id()
        self._project_data = None
        
        # Optional: validate inputs
        self._validate()

        print(f"Project {project_name} successfully created!")

    def _validate(self):
        """Private method to validate instance attributes."""
        if not isinstance(self._project_name, str):
            raise ValueError("Project name must be str")
    
    def __generate_id(self) -> str:
        """
        Private method to generate a secure identifier of the project.
        Returns a unique hexadecimal public identifier (7 hex chars + 2 hex chars).
        """

        char_num = 7
        max_value = 16 ** (char_num + 1)  # 16^8
        max_evals = int(1e4)
        count = 0

        file_path = ".quotesIdentifiers.txt"

        # Read existing numbers
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                used = {line.strip() for line in f}
        except FileNotFoundError:
            used = set()

        # Generate until unique
        while count < max_evals:
            secret = np.random.randint(0, max_value)
            if str(secret) not in used:
                break
            count += 1

        if count >= max_evals:
            raise Exception("Project initialization timed out during project identifier generation...")

        # Compute public hex ID (7 + 2 hex chars)
        block = max_value // 16
        part1 = secret % block
        part2 = secret // block

        public_number = f"{part1:0{char_num}x}{part2:02x}"

        # Store secret number for future uniqueness checks
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(str(secret) + "\n")

        return public_number

    def __checkId(self, public: str) -> dict:
        if not self._PUBLIC_ID_REGEX.match(public):
            return False

        file_path = ".quotesIdentifiers.txt"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                used = {line.strip() for line in f}
        except FileNotFoundError:
            return False

        char_num = 7
        max_value = 16 ** (char_num + 1)
        block = max_value // 16

        part1 = int(public[:char_num], 16)
        part2 = int(public[char_num:], 16)

        secret = part2 * block + part1

        res = {
            "status" : False,
            "secret" : -1
        }

        if str(secret) in used:
            res["status"] = True
            res["secret"] = secret

        return res



    @property
    def getName(self):
        """Getter for param1."""
        return self._project_name

    @project_name.setter
    def setProjectName(self, value):
        """Setter for param1 with optional validation."""
        self._project_name = value
        self._validate()

    def method(self, arg):
        """
        Example of a regular method.

        Args:
            arg (type): Description of the argument.

        Returns:
            type: Description of the return value.
        """
        # Do something with arg and instance variables
        return f"Processing {arg} with {self.param1}"
    
    def readFile(self, file_path : str) -> None:
        """
            Description. Creates the _project_data attributes based on a file
        """
        has_correct_extension = file_path.endswith('.csv') or file_path.endswith('.xlsx')
        has_pdf_extension = file_path.endswith('.pdf')

        if has_pdf_extension:
            pdf_path = file_path
            csv_path = file_path[:file_path.index('.pdf')] + ".csv"

            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[0]

                # pdfplumber often detects multiple partial tables, so we combine them
                tables = page.extract_tables()

                # Flattening all extracted tables into one
                cleaned_rows = []
                for tbl in tables:
                    for row in tbl:
                        cleaned_rows.append([cell.strip() if cell else "" for cell in row])

            # Write CSV
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(cleaned_rows)

        if has_correct_extension:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            data = data.rename(columns = {col : col.strip().lower().replace(' ', '_')})
            self._project_data = data

    @staticmethod
    def helper_function(x, y):
        """
        Example static utility method.
        """
        return x + y

    def __repr__(self):
        """Unambiguous string representation."""
        return f"QuoteProject <{_project_id}>"
