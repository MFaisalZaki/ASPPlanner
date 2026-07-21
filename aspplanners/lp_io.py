"""Generic, encoding-agnostic ASP program I/O.

This layer knows nothing about planning or the PLASP fact vocabulary: it is the
``ASPTerm`` identity base plus a thin classification/round-trip wrapper over
clingo's AST (`parse_lp`/`dump_lp`). Any encoding that reads or writes ``.lp``
text can build on it; the PLASP fact builders in ``aspplanners.plasp.facts`` are
one such consumer.
"""

from typing import IO, Iterable, List, Union

from clingo import ast as clingo_ast


class ASPTerm:
    """Base for every ASP fact/term wrapper: identity is the rendered string,
    so sets of terms deduplicate on their final ASP text."""

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, value):
        return str(self) == str(value)


# ---------------------------------------------------------------------------
# Reading and writing .lp files as ASPTerm objects
# ---------------------------------------------------------------------------

class ASPStatement(ASPTerm):
    """A statement parsed from ASP text, wrapping its clingo AST node.

    ``str()`` renders it back to valid (clingo-normalized) ASP syntax:
    comments are discarded, whitespace is normalized, and body literals are
    separated by ';'. The underlying ``clingo.ast.AST`` stays available as
    ``self.node`` for structural inspection.
    """

    def __init__(self, node):
        self.node = node

    def __str__(self):
        return str(self.node)


class ASPFact(ASPStatement):
    """A body-less rule: ``head.``"""

    @property
    def head(self) -> str:
        return str(self.node.head)


class ASPRule(ASPStatement):
    """``head :- body.``"""

    @property
    def head(self) -> str:
        return str(self.node.head)

    @property
    def body(self) -> List[str]:
        return [str(b) for b in self.node.body]


class ASPConstraint(ASPStatement):
    """An integrity constraint: ``:- body.``

    clingo's AST renders these as ``#false :- body.``; str() keeps the
    conventional head-less spelling.
    """

    @property
    def body(self) -> List[str]:
        return [str(b) for b in self.node.body]

    def __str__(self):
        return f":- {'; '.join(self.body)}."


class ASPWeakConstraint(ASPStatement):
    """``:~ body. [weight@priority, terms]``"""

    @property
    def body(self) -> List[str]:
        return [str(b) for b in self.node.body]


class ASPDirective(ASPStatement):
    """Any non-rule statement: #program, #show, #defined, #external,
    #script, #const, ..."""


def _wrap_statement(node) -> ASPStatement:
    if node.ast_type == clingo_ast.ASTType.Rule:
        head = node.head
        is_constraint = (head.ast_type == clingo_ast.ASTType.Literal
                         and head.atom.ast_type == clingo_ast.ASTType.BooleanConstant
                         and not head.atom.value)
        if is_constraint:
            return ASPConstraint(node)
        if len(node.body) == 0:
            return ASPFact(node)
        return ASPRule(node)
    if node.ast_type == clingo_ast.ASTType.Minimize:
        return ASPWeakConstraint(node)
    return ASPDirective(node)


def _is_base_program_node(node) -> bool:
    return (node.ast_type == clingo_ast.ASTType.Program
            and node.name == 'base' and len(node.parameters) == 0)


def parse_lp(text: str) -> List[ASPStatement]:
    """Parse ASP program text into a list of ASPStatement terms.

    clingo always emits an implicit ``#program base.`` as the first node
    (duplicating an explicit one); it is dropped so statements before any
    #program directive stay in the implicit base part and parse -> dump ->
    parse is a fixpoint.
    """
    nodes = []
    clingo_ast.parse_string(text, nodes.append)
    if nodes and _is_base_program_node(nodes[0]):
        nodes = nodes[1:]
    return [_wrap_statement(n) for n in nodes]


def parse_lp_file(path) -> List[ASPStatement]:
    """Parse an .lp file into a list of ASPStatement terms."""
    with open(path, 'r') as f:
        return parse_lp(f.read())


def dump_lp(terms: Iterable, destination: Union[str, IO[str]]) -> None:
    """Write ASP terms to `destination` (a file path or file-like object),
    one statement per line.

    Accepts anything that renders to ASP text with str(): parsed
    ASPStatement objects, the PLASP fact-builder ASPTerm classes (which may
    render to several lines), or plain strings.
    """
    text = '\n'.join(str(t) for t in terms) + '\n'
    if hasattr(destination, 'write'):
        destination.write(text)
    else:
        with open(destination, 'w') as f:
            f.write(text)
