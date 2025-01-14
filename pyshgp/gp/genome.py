"""The :mod:`genome` module defines the ``Genome`` type and provides genome translation, spawning and simplification.

A ``Genome`` is a persistent collection of gene Atoms (any `Atom` that isn't a ``CodeBlock``). It can be translated
into a ``CodeBlock``.

The ``GeneSpawner`` is a factory capable of generating random genes and random genomes. It is used for initializing a
population as well producing new genes used by variation operators (i.e. mutation).

The genome simplification process is useful for removing superfluous genes from a genome without negatively impacting
the behavior of the program produced by the genome. This process has many benefits including: improving generalization,
shrinking the size of the serialized solution, and in some cases making the program easier to explain.

"""
from __future__ import annotations

import random
from enum import Enum
from typing import Sequence, Union, Any, Callable, Tuple

import numpy as np
from pyrsistent import PRecord, field, CheckedPVector, l

from pyshgp.gp.evaluation import Evaluator
from pyshgp.push.instruction_set import InstructionSet
from pyshgp.push.program import ProgramSignature, Program
from pyshgp.push.atoms import Atom, CodeBlock, Closer, Literal, InstructionMeta, Input
from pyshgp.push.type_library import infer_literal
from pyshgp.tap import tap
from pyshgp.utils import DiscreteProbDistrib
from itertools import combinations


class Opener(PRecord):
    """Marks the start of one or more CodeBlock."""

    count = field(type=int, mandatory=True)

    def dec(self) -> Opener:
        """Create an ``Opener`` with ``count`` decremented."""
        return Opener(count=self.count - 1)


def _has_opener(seq: Sequence) -> bool:
    for el in seq:
        if isinstance(el, Opener):
            return True
    return False


class Genome(CheckedPVector):
    """A linear sequence of genes (aka any atom that isn't a ``CodeBlock``).

    PyshGP uses the Plushy genome representation.

    See: http://gpbib.cs.ucl.ac.uk/gp-html/Spector_2019_GPTP.html

    """

    __type__ = Atom
    __invariant__ = lambda a: (not isinstance(a, CodeBlock), 'CodeBlock')


def genome_to_code(genome: Genome) -> CodeBlock:
    """Translate into nested CodeBlocks.

    These CodeBlocks can be considered the Push program representation of
    the Genome which can be executed by a PushInterpreter and evaluated
    by an Evaluator.

    """
    plushy_buffer = l()
    for atom in genome[::-1]:
        if isinstance(atom, InstructionMeta) and atom.code_blocks > 0:
            plushy_buffer = plushy_buffer.cons(Opener(count=atom.code_blocks))
        plushy_buffer = plushy_buffer.cons(atom)

    push_buffer = []
    while True:
        # If done with plush but unclosed opens, recur with one more close.
        if len(plushy_buffer) == 0 and _has_opener(push_buffer):
            plushy_buffer = plushy_buffer.cons(Closer())
        # If done with plush and all opens closed, return push.
        elif len(plushy_buffer) == 0:
            return CodeBlock(push_buffer)
        else:
            atom = plushy_buffer.first
            plushy_buffer = plushy_buffer.rest
            # If next instruction is a close, and there is an open.
            if isinstance(atom, Closer) and _has_opener(push_buffer):
                ndx, opener = [(ndx, el) for ndx, el in enumerate(push_buffer) if isinstance(el, Opener)][-1]
                post_open = push_buffer[ndx + 1:]
                pre_open = push_buffer[:ndx]
                push_buffer = pre_open + [CodeBlock(post_open)]
                if opener.count > 1:
                    opener = opener.dec()
                    push_buffer.append(opener)
            # If next instruction is a close, and there is no open, ignore it.
            # If next instruction is not a close.
            elif not isinstance(atom, Closer):
                push_buffer.append(atom)


class GeneTypes(Enum):
    """An ``Enum`` denoting the different types of genes that can appear in a Genome."""

    INPUT = 1
    INSTRUCTION = 2
    CLOSE = 3
    LITERAL = 4
    ERC = 5


class GeneSpawner:
    """A factory of random Genes (Atoms) and Genomes.

    When  spawning a random gene, the result can be one of three types of Atoms.
    An Instruction, a Closer, or a Literal. If the Atom is a Literal, it may
    be one of the supplied Literals, or it may be the result of running one of
    the Ephemeral Random Constant generators.

    Reference for ERCs:
    "A field guide to genetic programming", Section 3.1
    Riccardo Poli and William B. Langdon and Nicholas Freitag McPhee,
    http://www.gp-field-guide.org.uk/

    Attributes
    ----------
    n_input : int
        Number of input instructions that could appear the genomes.
    instruction_set : pyshgp.push.instruction_set.InstructionSet
        InstructionSet containing instructions to use when spawning genes and
        genomes.
    literals : Sequence[pyshgp.push.instruction_set.atoms.Literal]
        A list of Literal objects to pull from when spawning genes and genomes.
    erc_generator : Sequence[Callable]
        A list of functions (aka Ephemeral Random Constant generators). When one
        of these functions is called, the output is placed in a Literal and
        returned as the spawned gene.
    distribution : pyshgp.utils.DiscreteProbDistrib
        A probability distribution describing how frequently to produce
        Instructions, Closers, Literals, and ERCs.

    """

    def __init__(self,
                 n_inputs: int,
                 instruction_set: Union[InstructionSet, str],
                 literals: Sequence[Any],
                 erc_generators: Sequence[Callable],
                 distribution: DiscreteProbDistrib = "proportional"):
        self.n_inputs = n_inputs
        self.erc_generators = erc_generators

        self.instruction_set = instruction_set
        if self.instruction_set == "core":
            self.instruction_set = InstructionSet(register_core=True)
        self.type_library = self.instruction_set.type_library
        self.literals = [lit if isinstance(lit, Literal) else infer_literal(lit, self.type_library) for lit in literals]

        if distribution == "proportional":
            self.distribution = (
                DiscreteProbDistrib()
                .add(GeneTypes.INPUT, self.n_inputs)
                .add(GeneTypes.INSTRUCTION, len(self.instruction_set))
                .add(GeneTypes.CLOSE, sum([i.code_blocks for i in self.instruction_set.values()]))
                .add(GeneTypes.LITERAL, len(literals))
                .add(GeneTypes.ERC, len(erc_generators))
            )
        else:
            self.distribution = distribution

    def random_input(self) -> Input:
        """Return a random ``Input``.

        Returns
        -------
        pyshgp.push.atoms.Input

        """
        return Input(input_index=np.random.randint(self.n_inputs))

    def random_instruction(self) -> InstructionMeta:
        """Return a random Instruction from the InstructionSet.

        Returns
        -------
        pyshgp.push.atoms.InstructionMeta
            A randomly selected Literal.

        """
        i = np.random.choice(list(self.instruction_set.values()))
        return InstructionMeta(name=i.name, code_blocks=i.code_blocks)

    def random_literal(self) -> Literal:
        """Return a random Literal from the set of Literals.

        Returns
        -------
        pyshgp.push.atoms.Literal
            A randomly selected Literal.

        """
        lit = np.random.choice(self.literals)
        if not isinstance(lit, Literal):
            lit = infer_literal(lit, self.type_library)
        return lit

    def random_erc(self) -> Literal:
        """Materialize a random ERC generator into a Literal and return it.

        Returns
        -------
        pyshgp.push.atoms.Literal
            A Literal whose value comes from running a ERC generator function.

        """
        erc_value = np.random.choice(self.erc_generators)()
        if not isinstance(erc_value, Literal):
            erc_value = infer_literal(erc_value, self.type_library)
        return erc_value

    def random_gene(self) -> Atom:
        """Return a random Atom based on the GenomeSpawner's distribution.

        Returns
        -------
        pyshgp.push.atoms.Atom
            An random Atom. Either an Instruction, Closer, or Literal.

        """
        atom_type = self.distribution.sample()
        if atom_type is GeneTypes.INPUT:
            return self.random_input()
        elif atom_type is GeneTypes.INSTRUCTION:
            return self.random_instruction()
        elif atom_type is GeneTypes.CLOSE:
            return Closer()
        elif atom_type is GeneTypes.LITERAL:
            return self.random_literal()
        elif atom_type is GeneTypes.ERC:
            return self.random_erc()
        else:
            raise ValueError("GenomeSpawner distribution bad atom type {t}".format(t=str(atom_type)))

    def spawn_genome(self, size: Union[int, Sequence[int]]) -> Genome:
        """Return a random Genome based on the GenomeSpawner's distribution.

        The genome will contain the specified number of Atoms if size is an
        integer. If size is a pair of integers, the genome will be of a random
        size in the range of the two integers.

        Parameters
        ----------
        size
            The resulting genome will contain this many Atoms if size is an
            integer. If size is a pair of integers, the genome will be of a random
            size in the range of the two integers.

        Returns
        -------
        pyshgp.gp.genome.Genome
            A Genome with random contents of a given size.

        """
        if isinstance(size, Sequence):
            size = np.random.randint(size[0], size[1]) + 1
        return Genome([self.random_gene() for _ in range(size)])


class GenomeSimplifier:
    """Simplifies a genome while preserving, or improving, its error.

    Genomes, and Push programs, can contain superfluous Push code. This extra
    code often has no effect on the program behavior, but occasionally it can
    introduce subtle errors or behaviors that is not covered by the training
    cases. Removing the superfluous code makes genomes (and thus programs)
    smaller and easier to understand. More importantly, simplification can
    improve the generalization of the given genome/program.

    The process of genome simplification is iterative and closely resembles
    simple hill climbing. For each iteration, the simplifier will randomly
    select a small number of random genes to remove. The Genome is re-evaluated
    and if its error gets worse, the change is reverted. After repeating this
    for some number of steps, the resulting genome will be the same size or
    smaller while containing the same (or better) error value.

    Reference:
    "Improving generalization of evolved programs through automatic simplification"
    Thomas Helmuth, Nicholas Freitag McPhee, Edward Pantridge, and Lee Spector. 2017.
    In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '17).
    ACM, New York, NY, USA, 937-944. DOI: https://doi.org/10.1145/3071178.3071330

    See: https://dl.acm.org/citation.cfm?id=3071178.3071330

    """

    # @TODO: Add noop swaps to simplification.

    def __init__(self,
                 evaluator: Evaluator,
                 program_signature: ProgramSignature):
        self.evaluator = evaluator
        self.program_signature = program_signature

    def _remove_rand_genes(self, genome: Genome, include_one: bool = False) -> Genome:
        # @todo DRY with deletion variation operator.
        gn = genome
        n_genes_to_remove = min(np.random.randint(1 if include_one else 2, 4), len(genome) - 1)
        ndx_of_genes_to_remove = np.random.choice(np.arange(len(gn)), n_genes_to_remove, replace=False)
        ndx_of_genes_to_remove[::-1].sort()
        print('Trying to remove', ndx_of_genes_to_remove)
        for ndx in ndx_of_genes_to_remove:
            gn = gn.delete(ndx)
        return gn

    def _errors_of_genome(self, genome: Genome) -> np.ndarray:
        cb = genome_to_code(genome)
        program = Program(code=cb, signature=self.program_signature)
        return self.evaluator.evaluate(program)

    @tap
    def _step(self, genome: Genome, errors_to_beat: np.ndarray, include_one: bool = False) -> Tuple[Genome, np.ndarray]:
        new_gn = self._remove_rand_genes(genome, include_one)
        new_errs = self._errors_of_genome(new_gn)
        if np.sum(new_errs) <= np.sum(errors_to_beat):
            print('simplified to', np.sum(new_errs), new_errs)
            return new_gn, new_errs
        return genome, errors_to_beat

    @tap
    def _step_sequential(self, genome: Genome, errors_to_beat: np.ndarray, ndx_of_genes_to_remove, printing) -> Tuple[Genome, np.ndarray]:
        new_gn = genome
        for ndx in reversed(ndx_of_genes_to_remove):
            new_gn = new_gn.delete(ndx)

        new_errs = self._errors_of_genome(new_gn)
        if printing:
            print('Error of that attempt was', np.sum(new_errs), new_errs)
        if np.sum(new_errs) <= np.sum(errors_to_beat):
            return new_gn, new_errs
        return genome, errors_to_beat

    @tap
    def simplify(self,
                 genome: Genome,
                 original_errors: np.ndarray,
                 steps: int = 2000) -> Tuple[Genome, np.ndarray]:
        """Simplify the given genome while maintaining error.

        Parameters
        ----------
        genome
            The Genome to simplify.
        original_errors
            Error vector of the genome to simplify.
        steps
            Number of simplification iterations to perform. Default is 2000.

        Returns
        -------
        pyshgp.gp.genome.Genome
            The shorter Genome that expresses the same computation.

        """
        gn = genome
        errs = original_errors
        checked_everything = False
        num_steps_taken = 0
        previous_step_found = 0
        original_length = len(gn)
        printing = True  # if random.random() > 0.99 else False
        if printing:
            print('Tasked with simplifying, starting with removing 1 at a time. Initial error is:', np.sum(errs), errs)
        while not checked_everything:
            if printing:
                print('Genome is', len(gn), 'long')
            previous_len = len(gn)
            for step in range(len(gn)):
                n_genes_to_remove = [(step + previous_step_found) % len(gn)]
                if printing:
                    print('Trying to remove item', (step + previous_step_found) % len(gn))
                gn, errs = self._step_sequential(gn, errs, n_genes_to_remove, printing)
                num_steps_taken += 1
                if not previous_len == len(gn):
                    previous_step_found = step + previous_step_found
                    if printing:
                        print('new program', genome_to_code(gn).pretty_str())
                    break
            else:
                checked_everything = True
        if printing:
            print('Tried to simplify by removing one gene and removed', original_length - len(gn), 'genes')

        if len(gn) <= 12:
            gn, errs = self.combination_simplify(gn, errs, printing)
        else:
            include_one = False
            start_gn = len(gn)
            for step in range(steps - num_steps_taken):
                gn, errs = self._step(gn, errs, include_one)
                if len(gn) < start_gn:
                    include_one = True
                if len(gn) == 1:
                    break
                if len(gn) < 12:
                    gn, errs = self.combination_simplify(gn, errs, printing)
                    break
        if printing:
            print('new program', genome_to_code(gn).pretty_str())
        return gn, errs

    def combination_simplify(self, gn, errs, printing):
        new_length = len(gn)
        if printing:
            print('Initiating combination simplify at', new_length)
        checked_everything = False
        while not checked_everything:
            for i in range(2, min(len(gn), 5)):
                n_genes_to_remove_list = combinations([x for x in range(len(gn))], i)
                found_simplification = False
                previous_len = len(gn)
                for n_genes_to_remove in n_genes_to_remove_list:
                    if printing:
                        print('Trying to remove genome indices:', n_genes_to_remove)
                    gn, errs = self._step_sequential(gn, errs, n_genes_to_remove, printing)

                    if not previous_len == len(gn):
                        if printing:
                            print('new program', genome_to_code(gn).pretty_str())
                        found_simplification = True
                        break
                if found_simplification:
                    break
            else:
                checked_everything = True
        if printing:
            print('Tried to simplify by removing combinations of genes and removed', new_length - len(gn), 'genes')
        return gn, errs

    @tap
    def simplify_quick(self,
                 genome: Genome,
                 original_errors: np.ndarray) -> Tuple[Genome, np.ndarray]:
        """Simplify the given genome while maintaining error.

        Parameters
        ----------
        genome
            The Genome to simplify.
        original_errors
            Error vector of the genome to simplify.
        steps
            Number of simplification iterations to perform. Default is 2000.

        Returns
        -------
        pyshgp.gp.genome.Genome
            The shorter Genome that expresses the same computation.

        """
        gn = genome
        errs = original_errors
        checked_everything = False

        while not checked_everything:
            previous_len = len(gn)
            for step in range(len(gn) - 1):
                n_genes_to_remove = [step]
                gn, errs = self._step_sequential(gn, errs, n_genes_to_remove, True)
                if not previous_len == len(gn):
                    break
            else:
                checked_everything = True
        return gn, errs
