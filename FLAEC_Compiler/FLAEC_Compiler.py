"""
FLAEC - AI-Driven Federated Learning Analytics Expression Compiler
CSA1404 - Compiler Design

Complete Python prototype implementing:
- Lexical analysis
- Token generation
- Lexical error detection
- LL(1)-friendly grammar
- Recursive-descent parsing
- Arithmetic precedence
- Parentheses
- Parse-tree generation
- Test cases
- Interactive compiler menu

Grammar:
    S  -> ID ASSIGN E
    E  -> T E'
    E' -> ADD_OP T E' | epsilon
    T  -> F T'
    T' -> MUL_OP F T' | epsilon
    F  -> LPAREN E RPAREN | ID | NUMBER
"""

from dataclasses import dataclass
from enum import Enum
import re


# ============================================================
# TOKEN DEFINITIONS
# ============================================================

class TokenType(Enum):
    IDENTIFIER = "IDENTIFIER"
    NUMBER = "NUMBER"
    ASSIGN = "ASSIGN"
    ADD_OP = "ADD_OP"
    MUL_OP = "MUL_OP"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    EOF = "EOF"


@dataclass
class Token:
    token_type: TokenType
    value: str
    position: int

    def __str__(self):
        return f"{self.token_type.value}({self.value})"


# ============================================================
# COMPILER ERRORS
# ============================================================

class CompilerError(Exception):
    pass


class LexicalError(CompilerError):
    pass


class FLAECSyntaxError(CompilerError):
    pass


# ============================================================
# LEXICAL ANALYZER
# ============================================================

class Lexer:
    IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    NUMBER = re.compile(r"(?:[0-9]+\.[0-9]+|[0-9]+)")

    def __init__(self, source):
        self.source = source
        self.position = 0
        self.tokens = []

    def tokenize(self):
        while self.position < len(self.source):

            ch = self.source[self.position]

            # Ignore whitespace
            if ch.isspace():
                self.position += 1
                continue

            # Identifier
            match = self.IDENTIFIER.match(
                self.source, self.position
            )

            if match:
                value = match.group()

                self.tokens.append(
                    Token(
                        TokenType.IDENTIFIER,
                        value,
                        self.position
                    )
                )

                self.position = match.end()
                continue

            # Number
            match = self.NUMBER.match(
                self.source, self.position
            )

            if match:
                value = match.group()

                self.tokens.append(
                    Token(
                        TokenType.NUMBER,
                        value,
                        self.position
                    )
                )

                self.position = match.end()
                continue

            # Single-character tokens
            token_map = {
                "=": TokenType.ASSIGN,
                "+": TokenType.ADD_OP,
                "-": TokenType.ADD_OP,
                "*": TokenType.MUL_OP,
                "/": TokenType.MUL_OP,
                "(": TokenType.LPAREN,
                ")": TokenType.RPAREN
            }

            if ch in token_map:

                self.tokens.append(
                    Token(
                        token_map[ch],
                        ch,
                        self.position
                    )
                )

                self.position += 1
                continue

            # Illegal symbol
            raise LexicalError(
                f"Illegal symbol '{ch}' "
                f"at position {self.position}"
            )

        self.tokens.append(
            Token(
                TokenType.EOF,
                "$",
                len(self.source)
            )
        )

        return self.tokens


# ============================================================
# PARSE TREE
# ============================================================

@dataclass
class ParseNode:

    name: str
    value: str = ""
    children: list = None

    def __post_init__(self):

        if self.children is None:
            self.children = []

    def add(self, child):
        self.children.append(child)

    def pretty(self, level=0):

        indentation = "  " * level

        label = self.name

        if self.value:
            label += f": {self.value}"

        lines = [indentation + label]

        for child in self.children:
            lines.append(
                child.pretty(level + 1)
            )

        return "\n".join(lines)


# ============================================================
# LL(1) / RECURSIVE-DESCENT PARSER
# ============================================================

class Parser:

    def __init__(self, tokens):

        self.tokens = tokens
        self.index = 0

    @property
    def current(self):

        return self.tokens[self.index]

    def advance(self):

        token = self.current
        self.index += 1

        return token

    def match(self, expected_type):

        if self.current.token_type == expected_type:
            return self.advance()

        raise FLAECSyntaxError(
            f"Expected {expected_type.value}, "
            f"found '{self.current.value}' "
            f"at position {self.current.position}"
        )

    # S -> ID ASSIGN E
    def parse(self):

        root = ParseNode("S")

        identifier = self.match(
            TokenType.IDENTIFIER
        )

        root.add(
            ParseNode("ID", identifier.value)
        )

        assignment = self.match(
            TokenType.ASSIGN
        )

        root.add(
            ParseNode("ASSIGN", assignment.value)
        )

        root.add(self.parse_E())

        if self.current.token_type != TokenType.EOF:

            raise FLAECSyntaxError(
                f"Unexpected token "
                f"'{self.current.value}' "
                f"at position "
                f"{self.current.position}"
            )

        return root

    # E -> T E'
    def parse_E(self):

        node = ParseNode("E")

        node.add(self.parse_T())

        while self.current.token_type == TokenType.ADD_OP:

            operator = self.advance()

            node.add(
                ParseNode(
                    "ADD_OP",
                    operator.value
                )
            )

            node.add(self.parse_T())

        return node

    # T -> F T'
    def parse_T(self):

        node = ParseNode("T")

        node.add(self.parse_F())

        while self.current.token_type == TokenType.MUL_OP:

            operator = self.advance()

            node.add(
                ParseNode(
                    "MUL_OP",
                    operator.value
                )
            )

            node.add(self.parse_F())

        return node

    # F -> ( E ) | ID | NUMBER
    def parse_F(self):

        node = ParseNode("F")

        if self.current.token_type == TokenType.LPAREN:

            left = self.advance()

            node.add(
                ParseNode(
                    "LPAREN",
                    left.value
                )
            )

            node.add(self.parse_E())

            right = self.match(
                TokenType.RPAREN
            )

            node.add(
                ParseNode(
                    "RPAREN",
                    right.value
                )
            )

            return node

        if self.current.token_type == TokenType.IDENTIFIER:

            token = self.advance()

            node.add(
                ParseNode(
                    "ID",
                    token.value
                )
            )

            return node

        if self.current.token_type == TokenType.NUMBER:

            token = self.advance()

            node.add(
                ParseNode(
                    "NUMBER",
                    token.value
                )
            )

            return node

        raise FLAECSyntaxError(
            "Expected identifier, number, "
            "or '(' but found "
            f"'{self.current.value}' "
            f"at position "
            f"{self.current.position}"
        )


# ============================================================
# FLAEC COMPILER
# ============================================================

class FLAECCompiler:

    def compile(self, expression):

        print("\n" + "=" * 70)
        print("FLAEC COMPILER")
        print("=" * 70)

        print("\nInput:")
        print(expression)

        # ----------------------------------------------------
        # Stage 1: Lexical Analysis
        # ----------------------------------------------------

        print("\n[1] LEXICAL ANALYSIS")
        print("-" * 40)

        try:

            lexer = Lexer(expression)
            tokens = lexer.tokenize()

            print("Status: SUCCESS")

            print("\nToken Stream:")

            for token in tokens:
                print(" ", token)

        except LexicalError as error:

            print("Status: FAILED")
            print("Error:", error)

            return False

        # ----------------------------------------------------
        # Stage 2: Syntax Analysis
        # ----------------------------------------------------

        print("\n[2] SYNTAX ANALYSIS")
        print("-" * 40)

        try:

            parser = Parser(tokens)
            tree = parser.parse()

            print("Status: SUCCESS")
            print("Expression accepted.")

        except FLAECSyntaxError as error:

            print("Status: FAILED")
            print("Error:", error)

            return False

        # ----------------------------------------------------
        # Stage 3: Parse Tree
        # ----------------------------------------------------

        print("\n[3] PARSE TREE")
        print("-" * 40)

        print(tree.pretty())

        # ----------------------------------------------------
        # Final Result
        # ----------------------------------------------------

        print("\n[4] FINAL RESULT")
        print("-" * 40)

        print("COMPILATION SUCCESSFUL")
        print("STATUS: ACCEPTED")

        return True


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [

    (
        "TC01",
        "a = b + c",
        "Accepted"
    ),

    (
        "TC02",
        "a = b * c + d",
        "Accepted"
    ),

    (
        "TC03",
        "a = (b + c) * d",
        "Accepted"
    ),

    (
        "TC04",
        "aggregationScore = "
        "(accuracy * clientWeight) + "
        "(dataQuality * participationRate)",
        "Accepted"
    ),

    (
        "TC05",
        "a = b + c / d",
        "Accepted"
    ),

    (
        "TC06",
        "a = (b + c",
        "Rejected"
    ),

    (
        "TC07",
        "a = b + * c",
        "Rejected"
    ),

    (
        "TC08",
        "a = b @ c",
        "Rejected"
    ),

    (
        "TC09",
        "a = 2abc + c",
        "Rejected"
    ),

    (
        "TC10",
        "a = b /",
        "Rejected"
    ),

    (
        "TC11",
        "a = ((b+c)*d)",
        "Accepted"
    ),

    (
        "TC12",
        "a = b # c",
        "Rejected"
    ),

    (
        "TC13",
        "score = privacyPenalty / "
        "communicationCost + 0.95",
        "Accepted"
    ),

    (
        "TC14",
        "score = accuracy ** weight",
        "Rejected"
    ),

    (
        "TC15",
        "score = @accuracy + weight",
        "Rejected"
    )
]


def run_tests():

    print("\n" + "=" * 80)
    print("FLAEC TEST SUITE")
    print("=" * 80)

    passed = 0
    failed = 0

    for test_id, expression, expected in TEST_CASES:

        try:

            tokens = Lexer(expression).tokenize()

            Parser(tokens).parse()

            actual = "Accepted"

        except CompilerError:

            actual = "Rejected"

        status = (
            "PASS"
            if actual == expected
            else "FAIL"
        )

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(
            f"{test_id:<6}"
            f" Expected: {expected:<9}"
            f" Actual: {actual:<9}"
            f" {status}"
        )

    print("\n" + "-" * 80)

    print("Total Tests :", len(TEST_CASES))
    print("Passed      :", passed)
    print("Failed      :", failed)

    if failed == 0:
        print("Overall     : ALL TESTS PASSED")
    else:
        print("Overall     : SOME TESTS FAILED")


# ============================================================
# TOKEN SPECIFICATION
# ============================================================

def show_tokens():

    print("\n" + "=" * 80)
    print("TOKEN SPECIFICATION")
    print("=" * 80)

    specifications = [

        (
            "IDENTIFIER",
            r"[A-Za-z_][A-Za-z0-9_]*",
            "accuracy"
        ),

        (
            "NUMBER",
            r"[0-9]+(\.[0-9]+)?",
            "0.95"
        ),

        (
            "ASSIGN",
            "=",
            "="
        ),

        (
            "ADD_OP",
            r"\+|-",
            "+ / -"
        ),

        (
            "MUL_OP",
            r"\*|/",
            "* /"
        ),

        (
            "LPAREN",
            r"\(",
            "("
        ),

        (
            "RPAREN",
            r"\)",
            ")"
        ),

        (
            "WHITESPACE",
            r"[ \t\n]+",
            "ignored"
        )
    ]

    print(
        f"{'TOKEN':<15}"
        f"{'REGULAR EXPRESSION':<35}"
        f"EXAMPLE"
    )

    print("-" * 80)

    for token, regex, example in specifications:

        print(
            f"{token:<15}"
            f"{regex:<35}"
            f"{example}"
        )


# ============================================================
# GRAMMAR
# ============================================================

def show_grammar():

    print("\n" + "=" * 80)
    print("FLAEC LL(1)-FRIENDLY GRAMMAR")
    print("=" * 80)

    grammar = [

        "S  -> IDENTIFIER ASSIGN E",

        "E  -> T E'",

        "E' -> ADD_OP T E' | epsilon",

        "T  -> F T'",

        "T' -> MUL_OP F T' | epsilon",

        "F  -> LPAREN E RPAREN | IDENTIFIER | NUMBER"
    ]

    for rule in grammar:
        print(rule)

    print("\nPrecedence:")

    print("1. Parentheses")
    print("2. Multiplication / Division")
    print("3. Addition / Subtraction")


# ============================================================
# SAMPLE EXPRESSION
# ============================================================

def run_sample():

    sample = (
        "aggregationScore = "
        "(accuracy * clientWeight) + "
        "(dataQuality * participationRate) - "
        "privacyPenalty / communicationCost + "
        "fairnessBonus"
    )

    FLAECCompiler().compile(sample)


# ============================================================
# INTERACTIVE MODE
# ============================================================

def interactive_mode():

    compiler = FLAECCompiler()

    print("\n" + "=" * 80)
    print("FLAEC INTERACTIVE COMPILER")
    print("=" * 80)

    print(
        "\nType an expression or type 'exit'."
    )

    while True:

        expression = input("\nFLAEC> ").strip()

        if expression.lower() == "exit":

            print("Compiler closed.")
            break

        if not expression:

            print("Please enter an expression.")
            continue

        compiler.compile(expression)


# ============================================================
# MAIN MENU
# ============================================================

def main():

    while True:

        print("\n")
        print("=" * 80)
        print("AI-DRIVEN FEDERATED LEARNING ANALYTICS COMPILER")
        print("FLAEC - COMPILER DESIGN PROJECT")
        print("=" * 80)

        print("\n1. Compile custom expression")
        print("2. Run all test cases")
        print("3. Show token specification")
        print("4. Show grammar")
        print("5. Run project sample")
        print("6. Interactive compiler")
        print("7. Exit")

        choice = input(
            "\nEnter choice (1-7): "
        ).strip()

        if choice == "1":

            expression = input(
                "\nEnter FL Analytics Expression:\n"
            ).strip()

            if expression:
                FLAECCompiler().compile(
                    expression
                )
            else:
                print("No expression entered.")

        elif choice == "2":

            run_tests()

        elif choice == "3":

            show_tokens()

        elif choice == "4":

            show_grammar()

        elif choice == "5":

            run_sample()

        elif choice == "6":

            interactive_mode()

        elif choice == "7":

            print(
                "\nThank you for using FLAEC Compiler."
            )
            break

        else:

            print(
                "\nInvalid choice. "
                "Please select 1-7."
            )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()
