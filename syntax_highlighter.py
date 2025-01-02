#!/usr/bin/env python3

from PyQt5.QtGui import (
    QSyntaxHighlighter,
    QTextCharFormat,
    QColor,
    QFont
)
from PyQt5.QtCore import QRegExp

class CppSyntaxHighlighter(QSyntaxHighlighter):
    """
    A simple C++ syntax highlighter for QPlainTextEdit.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._highlight_rules = []

        # Create text formats
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#C586C0"))  # Purple hue
        keyword_format.setFontWeight(QFont.Bold)

        # C++ keywords
        keywords = [
            "alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit", "atomic_noexcept",
            "auto", "bitand", "bitor", "bool", "break", "case", "catch", "char", "char8_t", "char16_t", "char32_t",
            "class", "compl", "const", "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await",
            "co_return", "co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast", "else", "enum",
            "explicit", "export", "extern", "false", "final", "float", "for", "friend", "goto", "if", "inline", "int",
            "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq",
            "private", "protected", "public", "reflexpr", "register", "reinterpret_cast", "requires", "return",
            "short", "signed", "sizeof", "static", "static_assert", "static_cast", "struct", "switch", "synchronized",
            "template", "this", "thread_local", "throw", "true", "try", "typedef", "typeid", "typename", "union",
            "unsigned", "using", "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq"
        ]

        # Build rules for each keyword
        for word in keywords:
            pattern = QRegExp(r"\b" + word + r"\b")
            self._highlight_rules.append((pattern, keyword_format))

        # Class name format
        class_format = QTextCharFormat()
        class_format.setForeground(QColor("#4EC9B0"))  # Light green
        pattern = QRegExp(r"\bQ[A-Z]\w*\b")
        self._highlight_rules.append((pattern, class_format))

        # Single-line comment format
        single_line_comment_format = QTextCharFormat()
        single_line_comment_format.setForeground(QColor("#608B4E"))
        self._highlight_rules.append((QRegExp(r"//[^\n]*"), single_line_comment_format))

        # Multi-line comment format
        self._multi_line_comment_format = QTextCharFormat()
        self._multi_line_comment_format.setForeground(QColor("#608B4E"))

        # String format
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#CE9178"))
        self._highlight_rules.append((QRegExp(r"\".*\""), string_format))
        self._highlight_rules.append((QRegExp(r"\'.*\'"), string_format))

        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#B5CEA8"))
        self._highlight_rules.append((QRegExp(r"\b[0-9]+(\.[0-9]+)?\b"), number_format))

        # Multi-line comment delimiters
        self._comment_start = QRegExp(r"/\*")
        self._comment_end = QRegExp(r"\*/")

    def highlightBlock(self, text):
        # Apply standard rules
        for pattern, fmt in self._highlight_rules:
            index = pattern.indexIn(text, 0)
            while index >= 0:
                length = pattern.matchedLength()
                self.setFormat(index, length, fmt)
                index = pattern.indexIn(text, index + length)

        # Handle multi-line comments
        self.setCurrentBlockState(0)

        start_index = 0
        if self.previousBlockState() != 1:
            start_index = self._comment_start.indexIn(text)

        while start_index >= 0:
            end_index = self._comment_end.indexIn(text, start_index)
            if end_index == -1:
                self.setCurrentBlockState(1)
                comment_length = len(text) - start_index
            else:
                comment_length = end_index - start_index + self._comment_end.matchedLength()
            self.setFormat(start_index, comment_length, self._multi_line_comment_format)
            if end_index == -1:
                break
            else:
                start_index = self._comment_start.indexIn(text, start_index + comment_length)


class OutputHighlighter(QSyntaxHighlighter):
    """
    A simple highlighter for the output log. 
    Highlights lines containing 'error', 'warning', or 'info'.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

    def highlightBlock(self, text):
        lower_text = text.lower()
        if "error" in lower_text:
            error_format = QTextCharFormat()
            error_format.setForeground(QColor("#F44747"))  # Red
            self.setFormat(0, len(text), error_format)
        elif "warning" in lower_text:
            warn_format = QTextCharFormat()
            warn_format.setForeground(QColor("#FFA500"))  # Orange
            self.setFormat(0, len(text), warn_format)
        elif "info" in lower_text:
            info_format = QTextCharFormat()
            info_format.setForeground(QColor("#0FAF0F"))  # Green
            self.setFormat(0, len(text), info_format)
