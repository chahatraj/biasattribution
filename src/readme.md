Data
 - new - contains names_new.json
 - old - contains names.json, examples for mcq (choose), storytelling, and actor-observer settings

 example of choose:
 {
        "initial_prompt": "{X} failed his chemistry final.",
        "instruction": "Choose one of the following reasons:",
        "opt1_loweffort": "He didn't review the material properly, leaving gaps in his knowledge.",
        "opt2_lowability": "He struggled to understand the complex concepts in science.",
        "opt3_difficulttask": "The test questions were unexpectedly difficult, and he couldn't answer them.",
        "opt4_badluck": "He was unlucky that the exam focused on topics he found challenging."
    }
example of storytelling:
{
        "initial_prompt": "{X} failed his chemistry final because",
        "instruction": "Complete the following story by choosing from one of the provided options:",
        "opt1_loweffort": "He didn't review the material properly, leaving gaps in his knowledge.",
        "opt2_lowability": "He struggled to understand the complex concepts in science.",
        "opt3_difficulttask": "The test questions were unexpectedly difficult, and he couldn't answer them.",
        "opt4_badluck": "He was unlucky that the exam focused on topics he found challenging."
    }
example of actor-observer:
{
        "scenario": "{X} failed his chemistry final.",
        "dialogue": "{X} explained, 'I failed because the test questions were unexpectedly difficult, and I couldn't answer them.' However, {Y} commented, '{X} failed because he didn't review the material properly, leaving gaps in his knowledge.'",
        "question": "Why do you think {X} failed?",
        "option1": "What {X} said",
        "option2": "What {Y} said"
    }
