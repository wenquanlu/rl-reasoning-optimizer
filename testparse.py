from math_verify import LatexExtractionConfig, parse, verify

# gold_parsed = parse(
#     "\\boxed{\\begin{pmatrix} \\frac{5}{6} & \\frac{1}{3} & -\\frac{1}{6} \\\ \\frac{1}{3} & \\frac{1}{3} & \\frac{1}{3} \\\ -\\frac{1}{6} & \\frac{1}{3} & \\frac{5}{6} \\end{pmatrix}}",
#     extraction_mode="first_match",
# )

# from open_r1.utils.math_grader import is_latex_equal

# print(is_latex_equal("\\boxed{\\begin{pmatrix} \\frac{5.0}{6.0} & \\frac{1}{3} & -\\frac{1}{6} \\\ \\frac{1}{3} & \\frac{1}{3} & \\frac{1}{3} \\\ -\\frac{1}{6} & \\frac{1}{3} & \\frac{5}{6} \\end{pmatrix}}", "\\boxed{\\begin{pmatrix} \\frac{5}{6} & \\frac{1}{3} & -\\frac{1}{6} \\\ \\frac{1}{3} & \\frac{1}{3} & \\frac{1}{3} \\\ -\\frac{1}{6} & \\frac{1}{3} & \\frac{5}{6} \\end{pmatrix}}"))

gold_parsed = parse(
    "\\begin{pmatrix} \\frac{5}{6} & \\frac{1}{3} & -\\frac{1}{6} \\\ \\frac{1}{3} & \\frac{1}{3} & \\frac{1}{3} \\\ -\\frac{1}{6} & \\frac{1}{3} & \\frac{5}{6} \\end{pmatrix}"
)

print(gold_parsed)

gold_parsed = parse(
    "\\boxed{\\begin{pmatrix} \\frac{5}{6} & \\frac{1}{3} & -\\frac{1}{6} \\\ \\frac{1}{3} & \\frac{1}{3} & \\frac{1}{3} \\\ -\\frac{1}{6} & \\frac{1}{3} & \\frac{5}{6} \\end{pmatrix}}"
)
print(gold_parsed)

gold_parsed = parse("I think 2 or 3 is not a correct answer, I don't know")
print(gold_parsed)

gold_parsed = parse("the \\fbox{{123}} haha ")
print(gold_parsed)