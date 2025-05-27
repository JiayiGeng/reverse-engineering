"""
python -m blackboxes.languages
"""
import math
import json
import yaml
import random
from rich import print
from pathlib import Path
import numpy as np
import random
import re
from typing import Any, Dict, List
from tqdm import tqdm
import copy
from utils import load_data_from_yaml


class Languages:
    def __init__(self, config_path = None, language_name="language_1"):
        language_config = load_data_from_yaml(config_path) if config_path else load_data_from_yaml("configs/languages.yaml")
        language_config = language_config[language_name]
        self.language_rule = language_config['language_rule']
        self.description = language_config['description']
        
        self.RULE_TO_FUNC = {
            "An": self.an_func,
            "AB": self.ab_func,
            "ABn": self.abn_func,
            "AAA": self.aaa,
            "AAAA": self.aaaa,
            "AnBm": self.anbm,
            "GoldenMean": self.golden_mean,
            "Even": self.even_a_groups,
            "ApBAp": self.appba_p,
            "ApBApp": self.appba_pp,
            "AsBAsp": self.asba_sp,
            "CountA2": self.count_a2,
            "CountAEven": self.count_a_even,
            "aABb": self.aab_b,
            "AnBn": self.an_bn,
            "Dyck": self.dyck,
            "AnB2n": self.an_b2n,
            "AnCBn": self.an_cbn,
            "AnABn": self.an_abn,
            "ABnABAn": self.abn_aban,
            "AnBmCn": self.an_bm_cn,
            "AnBmA2n": self.an_bm_a2n,
            "AnBnC2n": self.an_bn_c2n,
            "AnBmCm": self.an_bm_cm,
            "AnBmCnpm": self.an_bm_cnpm,
            "AnBmCnm": self.an_bm_cnm,
            "AnBk": self.an_bk,
            "AnBmCmAn": self.an_bm_cm_an,
            "AnB2nC3n": self.an_b2n_c3n,
            "AnBnp1Cnp2": self.an_bnp1_cnp2,
            "AnUBn": self.an_u_bn,
            "AnUAnBn": self.an_u_anbn,
            "ABnUBAn": self.abn_u_ban,
            "XX": self.xx,
            "XXX": self.xxx,
            "XY": self.xy,
            "XXR": self.xxr,
            "XXI": self.xxi,
            "XXRI": self.xxri,
            "An2": self.an2,
            "AnBmCnDm": self.an_bm_cn_dm,
            "AnBmAnBm": self.an_bm_an_bm,
            "AnBmAnBmCCC": self.an_bm_an_bm_ccc,
            "AnBmAnBmCnDm": self.an_bm_an_bm_cn_dm,
            "A2en": self.a2en,
            "ABnen": self.ab_nen,
        }
        self.TEST_TO_FUNC = {
            "An": self.an_test,
            "AB": self.ab_test,
            "ABn": self.abn_test,
            "AAA": self.test_aaa,
            "AAAA": self.test_aaaa,
            "AnBm": self.test_anbm,
            "GoldenMean": self.test_golden_mean,
            "Even": self.test_even_a_groups,
            "ApBAp": self.test_appba_p,
            "ApBApp": self.test_appba_pp,
            "AsBAsp": self.test_asba_sp,
            "CountA2": self.test_count_a2,
            "CountAEven": self.test_count_a_even,
            "aABb": self.test_aab_b,
            "AnBn": self.test_an_bn,
            "Dyck": self.test_dyck,
            "AnB2n": self.test_an_b2n,
            "AnCBn": self.test_an_cbn,
            "AnABn": self.test_an_abn,
            "ABnABAn": self.test_abn_aban,
            "AnBmCn": self.test_an_bm_cn,
            "AnBmA2n": self.test_an_bm_a2n,
            "AnBnC2n": self.test_an_bn_c2n,
            "AnBmCm": self.test_an_bm_cm,
            "AnBmCnpm": self.test_an_bm_cnpm,
            "AnBmCnm": self.test_an_bm_cnm,
            "AnBk": self.test_an_bk,
            "AnBmCmAn": self.test_an_bm_cm_an,
            "AnB2nC3n": self.test_an_b2n_c3n,
            "AnBnp1Cnp2": self.test_an_bnp1_cnp2,
            "AnUBn": self.test_an_u_bn,
            "AnUAnBn": self.test_an_u_anbn,
            "ABnUBAn": self.test_abn_u_ban,
            "XX": self.test_xx,
            "XXX": self.test_xxx,
            "XY": self.test_xy,
            "XXR": self.test_xxr,
            "XXI": self.test_xxi,
            "XXRI": self.test_xxri,
            "An2": self.test_an2,
            "AnBmCnDm": self.test_an_bm_cn_dm,
            "AnBmAnBm": self.test_an_bm_an_bm,
            "AnBmAnBmCCC": self.test_an_bm_an_bm_ccc,
            "AnBmAnBmCnDm": self.test_an_bm_an_bm_cn_dm,
            "A2en": self.test_a2en,
            "ABnen": self.test_ab_nen,
        }
        self.language_func = self.RULE_TO_FUNC[self.language_rule]
        self.test_func = self.TEST_TO_FUNC[self.language_rule]

    def an_func(self):
        max_length = 100
        length = random.randint(1, max_length)
        return "".join(random.choices("A", k=length))
    
    def an_test(self, query):
        for char in query:
            if char != "A":
                return False
        return True
    
    def ab_func(self):
        max_length = 100
        length = random.randint(1, max_length)
        return "".join(random.choices("AB", k=length))
    
    def ab_test(self, query):
        pattern = r'^[AB]*$'
        match = re.match(pattern, query)
        return match is not None
    
    def abn_func(self):
        max_length = 100
        length = random.randint(1, max_length)
        return "".join(random.choices("AB", k=length))
    
    def abn_test(self, query):
        pattern = r'^[AB]*$'
        match = re.match(pattern, query)
        return match is not None
    
    def aaa(self):
        options = ['A', 'AA', 'AAA']
        return random.choice(options)

    def test_aaa(self, query):
        return query in {'A', 'AA', 'AAA'}

    def aaaa(self):
        options = ['A', 'AA', 'AAA', 'AAAA']
        return random.choice(options)

    def test_aaaa(self, query):
        return query in {'A', 'AA', 'AAA', 'AAAA'}

    def anbm(self, max_n=20, max_m=20):
        n = random.randint(0, max_n)
        m = random.randint(0, max_m)
        return 'A' * n + 'B' * m

    def test_anbm(self, query):
        pattern = r'^A*B*$'
        return re.fullmatch(pattern, query) is not None

    def golden_mean(self, max_length=20):
        result = []
        previous_char = None
        length = random.randint(0, max_length)
        for _ in range(length):
            if previous_char == 'A':
                char = 'B'
            else:
                char = random.choice(['A', 'B'])
            result.append(char)
            previous_char = char
        return ''.join(result)

    def test_golden_mean(self, query):
        return 'AA' not in query and set(query).issubset({'A', 'B'})

    def even_a_groups(self, max_length=20):
        result = []
        length = random.randint(0, max_length)
        i = 0
        while i < length:
            if random.choice([True, False]):
                # Insert 'B'
                result.append('B')
                i += 1
            else:
                # Calculate remaining length
                remaining = length - i
                if remaining < 2:
                    # Not enough space to insert even number of 'A's, insert 'B' instead
                    result.append('B')
                    i += 1
                    continue
                # Maximum number of even 'A's we can insert
                max_possible_a = remaining // 2
                # Ensure at least 1 pair of 'A's
                num_pairs = random.randint(1, max_possible_a)
                num_a = num_pairs * 2
                result.append('A' * num_a)
                i += num_a
        return ''.join(result)

    def test_even_a_groups(query):
        # Check that the string contains only 'A's and 'B's
        if not set(query).issubset({'A', 'B'}):
            return False
        
        # Use regex to find all groups of consecutive 'A's
        a_groups = re.findall(r'A+', query)
        for group in a_groups:
            if len(group) % 2 != 0:
                return False
        return True

    def appba_p(self, max_a=10):
        n = random.randint(1, max_a)
        m = random.randint(1, max_a)
        return 'A' * n + 'B' + 'A' * m

    def test_appba_p(self, query):
        pattern = r'^A+B A+$'
        return re.fullmatch(r'^A+B A+$', query) is not None

    def appba_pp(self, max_a=10, max_ba=10):
        n = random.randint(1, max_a)
        m = random.randint(1, max_ba)
        ba_sequences = ''.join(['BA' * random.randint(1, max_a) for _ in range(m)])
        return 'A' * n + ba_sequences

    def test_appba_pp(self, query):
        pattern = r'^A+(BA+)+$'
        return re.fullmatch(pattern, query) is not None

    def asba_sp(self):
        result = []
        # Start with A*
        num_a_start = random.randint(0, 10)
        result.append('A' * num_a_start)
        
        # Add one or more (B A*)
        num_sequences = random.randint(1, 5)
        for _ in range(num_sequences):
            result.append('B')
            num_a = random.randint(0, 10)
            result.append('A' * num_a)
        
        return ''.join(result)

    def test_asba_sp(self, query):
        pattern = r'^A*(BA*)+$'
        return re.fullmatch(pattern, query) is not None

    def count_a2(self, max_length=20):
        if max_length < 2:
            raise ValueError("max_length must be at least 2 to include at least two 'A's.")
        
        num_a = random.randint(2, max_length)
        num_b = random.randint(0, max_length - num_a)
        chars = ['A'] * num_a + ['B'] * num_b
        random.shuffle(chars)
        return ''.join(chars)

    def test_count_a2(self, query):
        """
        Tests whether the given string has at least two 'A's and consists only of 'A's and 'B's.
        
        Args:
            query (str): The string to be tested.
        
        Returns:
            bool: True if the string has at least two 'A's and contains only 'A' and 'B', False otherwise.
        """
        return query.count('A') >= 2 and set(query).issubset({'A', 'B'})


    def count_a_even(self, max_length=20):
        num_a = random.randint(0, max_length // 2) * 2  # Ensure even number of 'A's
        num_b = random.randint(0, max_length - num_a)
        chars = ['A'] * num_a + ['B'] * num_b
        random.shuffle(chars)
        return ''.join(chars)

    def test_count_a_even(self, query):
        return query.count('A') % 2 == 0 and set(query).issubset({'A', 'B'})

    def aab_b(self):
        middle_length = random.randint(1, 10)
        middle = ''.join(random.choice(['A', 'B']) for _ in range(middle_length))
        return f'a{middle}b'

    def test_aab_b(self, query):
        pattern = r'^a[AB]+b$'
        return re.fullmatch(pattern, query) is not None

    def an_bn(self, max_n=10):
        n = random.randint(0, max_n)
        return 'A' * n + 'B' * n

    def test_an_bn(self, query):
        match = re.fullmatch(r'^(A*)B*\1$', query)
        if match:
            num_a = len(match.group(1))
            num_b = len(query) - num_a
            return num_a == num_b
        return False

    def dyck(self, max_pairs=10):
        def generate(n):
            if n == 0:
                return ''
            result = '('
            n -= 1
            if n == 0:
                return '()'
            m = random.randint(0, n)
            return '(' + generate(m) + ')' + generate(n - m)
        
        num_pairs = random.randint(0, max_pairs)
        return generate(num_pairs)

    def test_dyck(self, query):
        stack = []
        for char in query:
            if char == '(':
                stack.append(char)
            elif char == ')':
                if not stack:
                    return False
                stack.pop()
            else:
                # Invalid character
                return False
        return not stack

    def an_b2n(self, max_n=10):
        n = random.randint(0, max_n)
        return 'A' * n + 'B' * (2 * n)

    def test_an_b2n(self, query):
        match = re.fullmatch(r'^(A*)B*$', query)
        if match:
            n = len(match.group(1))
            expected_b = 2 * n
            actual_b = len(query) - n
            return actual_b == expected_b
        return False

    def an_cbn(self, max_n=10):
        n = random.randint(0, max_n)
        return 'A' * n + 'C' + 'B' * n

    def test_an_cbn(self, query):
        pattern = r'^(A*)C(B*)$'
        match = re.fullmatch(pattern, query)
        if match:
            n_a = len(match.group(1))
            n_b = len(match.group(2))
            return n_a == n_b
        return False

    def an_abn(self, max_n=10):
        n = random.randint(0, max_n)
        return 'A' * n + 'AB' * n

    def test_an_abn(self, query):
        match = re.fullmatch(r'^(A*)(AB*)$', query)
        if match:
            n_a = len(match.group(1))
            ab_part = match.group(2)
            return ab_part == 'AB' * n_a
        return False

    def abn_aban(self, max_n=10):
        n = random.randint(0, max_n)
        return 'AB' * n + 'ABA' * n

    def test_abn_aban(self, query):
        n = query.count('AB')
        expected = 'AB' * n + 'ABA' * n
        return query == expected

    def an_bm_cn(self, max_n=10, max_m=10):
        n = random.randint(0, max_n)
        m = random.randint(0, max_m)
        return 'A' * n + 'B' * m + 'C' * n

    def test_an_bm_cn(self, query):
        pattern = r'^(A*)(B*)(C*)$'
        match = re.fullmatch(pattern, query)
        if match:
            n_a = len(match.group(1))
            m_b = len(match.group(2))
            n_c = len(match.group(3))
            return n_a == n_c
        return False

    def an_bm_a2n(self, max_n=10, max_m=10):
        n = random.randint(0, max_n)
        m = random.randint(0, max_m)
        return 'A' * n + 'B' * m + 'A' * (2 * n)

    def test_an_bm_a2n(self, query):
        pattern = r'^(A*)(B*)(A*)$'
        match = re.fullmatch(pattern, query)
        if match:
            n_a1 = len(match.group(1))
            m_b = len(match.group(2))
            n_a2 = len(match.group(3))
            return n_a2 == 2 * n_a1
        return False

    def an_bn_c2n(self, max_n=10):
        n = random.randint(0, max_n)
        return 'A' * n + 'B' * n + 'C' * (2 * n)

    def test_an_bn_c2n(self, query):
        pattern = r'^(A*)(B*)(C*)$'
        match = re.fullmatch(pattern, query)
        if match:
            n_a = len(match.group(1))
            n_b = len(match.group(2))
            n_c = len(match.group(3))
            return n_a == n_b and n_c == 2 * n_a
        return False

    def an_bm_cm(self, max_n=10, max_m=10):
        n = random.randint(0, max_n)
        m = random.randint(0, max_m)
        return 'A' * n + 'B' * m + 'C' * m

    def test_an_bm_cm(self, query):
        pattern = r'^(A*)(B*)(C*)$'
        match = re.fullmatch(pattern, query)
        if match:
            n_a = len(match.group(1))
            m_b = len(match.group(2))
            m_c = len(match.group(3))
            return m_b == m_c
        return False

    def an_bm_cnpm(self, max_n=10, max_m=10):
        n = random.randint(0, max_n)
        m = random.randint(0, max_m)
        return 'A' * n + 'B' * m + 'C' * (n + m)

    def test_an_bm_cnpm(self, query):
        pattern = r'^(A*)(B*)(C*)$'
        match = re.fullmatch(pattern, query)
        if match:
            n_a = len(match.group(1))
            m_b = len(match.group(2))
            c_c = len(match.group(3))
            return c_c == (n_a + m_b)
        return False

    def an_bm_cnm(self, max_n=10, max_m=10):
        n = random.randint(0, max_n)
        m = random.randint(0, max_m)
        return 'A' * n + 'B' * m + 'C' * (n * m)

    def test_an_bm_cnm(self, query):
        pattern = r'^(A*)(B*)(C*)$'
        match = re.fullmatch(pattern, query)
        if match:
            n_a = len(match.group(1))
            m_b = len(match.group(2))
            c_c = len(match.group(3))
            return c_c == (n_a * m_b)
        return False

    def an_bk(self, max_n=10, max_m=10):
        n = random.randint(0, max_n)
        m = random.randint(0, max_m)
        return 'A' * n + 'B' * (n + m)

    def test_an_bk(self, query):
        pattern = r'^(A*)(B*)$'
        match = re.fullmatch(pattern, query)
        if match:
            n_a = len(match.group(1))
            b_b = len(match.group(2))
            return b_b >= n_a
        return False

    def an_bm_cm_an(self, max_n=10, max_m=10):
        n = random.randint(0, max_n)
        m = random.randint(0, max_m)
        return 'A' * n + 'B' * m + 'C' * m + 'A' * n

    def test_an_bm_cm_an(self, query):
        pattern = r'^(A*)(B*)(C*)(A*)$'
        match = re.fullmatch(pattern, query)
        if match:
            n_a1 = len(match.group(1))
            m_b = len(match.group(2))
            m_c = len(match.group(3))
            n_a2 = len(match.group(4))
            return n_a1 == n_a2 and m_b == m_c
        return False

    def an_b2n_c3n(self, max_n=10):
        n = random.randint(0, max_n)
        return 'A' * n + 'B' * (2 * n) + 'C' * (3 * n)

    def test_an_b2n_c3n(self, query):
        pattern = r'^(A*)(B*)(C*)$'
        match = re.fullmatch(pattern, query)
        if match:
            n_a = len(match.group(1))
            n_b = len(match.group(2))
            n_c = len(match.group(3))
            return n_b == 2 * n_a and n_c == 3 * n_a
        return False

    def an_bnp1_cnp2(self, max_n=10):
        n = random.randint(0, max_n)
        return 'A' * n + 'B' * (n + 1) + 'C' * (n + 2)

    def test_an_bnp1_cnp2(self, query):
        pattern = r'^(A*)(B*)(C*)$'
        match = re.fullmatch(pattern, query)
        if match:
            n_a = len(match.group(1))
            n_b = len(match.group(2))
            n_c = len(match.group(3))
            return n_b == n_a + 1 and n_c == n_a + 2
        return False

    def an_u_bn(self, max_n=10):
        n = random.randint(0, max_n)
        choice = random.choice(['A', 'B'])
        return choice * n

    def test_an_u_bn(self, query):
        return re.fullmatch(r'^A+$', query) is not None or re.fullmatch(r'^B+$', query) is not None or query == ''

    def an_u_anbn(self, max_n=10):
        n = random.randint(0, max_n)
        choice = random.choice(['A_only', 'A_then_B'])
        if choice == 'A_only':
            return 'A' * n
        else:
            return 'A' * n + 'B' * n

    def test_an_u_anbn(self, query):
        if re.fullmatch(r'^A+$', query) is not None or query == '':
            return True
        pattern = r'^(A+)B+\1$'
        match = re.fullmatch(pattern, query)
        if match:
            n_a = len(match.group(1))
            n_b = len(query) - n_a
            return n_a == n_b
        return False

    def abn_u_ban(self, max_n=10):
        n = random.randint(0, max_n)
        choice = random.choice(['AB', 'BA'])
        return choice * n

    def test_abn_u_ban(self, query):
        if query == '':
            return True
        pattern_ab = r'^(AB)+$'
        pattern_ba = r'^(BA)+$'
        return re.fullmatch(pattern_ab, query) is not None or re.fullmatch(pattern_ba, query) is not None

    def xx(self, max_length=10):
        half_length = random.randint(0, max_length)
        base = ''.join(random.choices(['A', 'B'], k=half_length))
        return base * 2

    def test_xx(self, query):
        length = len(query)
        if length % 2 != 0:
            return False
        half = length // 2
        return query[:half] == query[half:]

    def xxx(self, max_length=10):
        third_length = random.randint(0, max_length)
        base = ''.join(random.choices(['A', 'B'], k=third_length))
        return base * 3

    def test_xxx(self, query):
        length = len(query)
        if length % 3 != 0:
            return False
        third = length // 3
        return query[:third] == query[third:2*third] == query[2*third:]

    def xy(self, max_length=10):
        length = random.randint(2, max_length)
        split_point = random.randint(1, length - 1)
        x_len = split_point
        # Generate Y
        y_len = length - split_point
        while True:
            X = ''.join(random.choices(['A', 'B'], k=x_len))
            Y = ''.join(random.choices(['A', 'B'], k=y_len))
            if X != Y:
                return X + Y

    def test_xy(self, query):
        for split_point in range(1, len(query)):
            X = query[:split_point]
            Y = query[split_point:]
            if X != Y:
                return True
        return False

    def xxr(self, max_length=10):
        x_len = random.randint(1, max_length)
        X = ''.join(random.choices(['A', 'B'], k=x_len))
        return X + X[::-1]

    def test_xxr(self, query):
        if len(query) % 2 != 0:
            return False
        half = len(query) // 2
        X = query[:half]
        XR = query[half:]
        return XR == X[::-1]

    def invert_string(self, s):
        return ''.join('B' if c == 'A' else 'A' for c in s)

    def xxi(self, max_length=10):
        x_len = random.randint(1, max_length)
        X = ''.join(random.choices(['A', 'B'], k=x_len))
        return X + self.invert_string(X)

    def test_xxi(self, query):
        if len(query) % 2 != 0:
            return False
        half = len(query) // 2
        X = query[:half]
        X_inv = query[half:]
        return X_inv == self.invert_string(X)

    def xxri(self, max_length=10):
        x_len = random.randint(1, max_length)
        X = ''.join(random.choices(['A', 'B'], k=x_len))
        return X + self.invert_string(X[::-1])

    def test_xxri(self, query):
        if len(query) % 2 != 0:
            return False
        half = len(query) // 2
        X = query[:half]
        second_half = query[half:]
        return second_half == self.invert_string(X[::-1])

    def an2(self, max_n=10):
        n = random.randint(0, max_n)
        length = n * n
        return 'A' * length

    def test_an2(self, query):
        if set(query) == set() or set(query) == {'A'}:
            # Check if length is a perfect square
            length = len(query)
            root = int(length**0.5)
            return root * root == length
        return False

    def an_bm_cn_dm(self, max_n=10, max_m=10):
        n = random.randint(0, max_n)
        m = random.randint(0, max_m)
        return 'A' * n + 'B' * m + 'C' * n + 'D' * m

    def test_an_bm_cn_dm(self, query):
        pattern = r'^(A*)(B*)(C*)(D*)$'
        match = re.fullmatch(pattern, query)
        if match:
            n_a = len(match.group(1))
            m_b = len(match.group(2))
            n_c = len(match.group(3))
            m_d = len(match.group(4))
            return n_a == n_c and m_b == m_d
        return False

    def an_bm_an_bm(self, max_n=10, max_m=10):
        n = random.randint(0, max_n)
        m = random.randint(0, max_m)
        return 'A' * n + 'B' * m + 'A' * n + 'B' * m

    def test_an_bm_an_bm(self, query):
        pattern = r'^(A*)(B*)(A*)(B*)$'
        match = re.fullmatch(pattern, query)
        if match:
            n_a1 = len(match.group(1))
            m_b1 = len(match.group(2))
            n_a2 = len(match.group(3))
            m_b2 = len(match.group(4))
            return n_a1 == n_a2 and m_b1 == m_b2
        return False

    def an_bm_an_bm_ccc(self, max_n=10, max_m=10):
        n = random.randint(0, max_n)
        m = random.randint(0, max_m)
        return 'A' * n + 'B' * m + 'A' * n + 'B' * m + 'CCC'

    def test_an_bm_an_bm_ccc(self, query):
        pattern = r'^(A*)(B*)(A*)(B*)CCC$'
        match = re.fullmatch(pattern, query)
        if match:
            n_a1 = len(match.group(1))
            m_b1 = len(match.group(2))
            n_a2 = len(match.group(3))
            m_b2 = len(match.group(4))
            return n_a1 == n_a2 and m_b1 == m_b2
        return False

    def an_bm_an_bm_cn_dm(self, max_n=10, max_m=10):
        n = random.randint(0, max_n)
        m = random.randint(0, max_m)
        return (
            'A' * n +
            'B' * m +
            'A' * n +
            'B' * m +
            'C' * n +
            'D' * m
        )

    def test_an_bm_an_bm_cn_dm(self, query):
        pattern = r'^(A*)(B*)(A*)(B*)(C*)(D*)$'
        match = re.fullmatch(pattern, query)
        if match:
            n_a1 = len(match.group(1))
            m_b1 = len(match.group(2))
            n_a2 = len(match.group(3))
            m_b2 = len(match.group(4))
            n_c = len(match.group(5))
            m_d = len(match.group(6))
            return n_a1 == n_a2 == n_c and m_b1 == m_b2 == m_d
        return False
    
    def a2en(self, max_n=10):
        # Since 2^10 = 1024, that's still not too large.
        n = random.randint(0, max_n)
        length = 2 ** n
        return 'A' * length

    def test_a2en(self, query):
        if set(query) == set() or set(query) == {'A'}:
            length = len(query)
            if length < 1:
                return False
            return (length & (length - 1)) == 0
        return False

    def ab_nen(self, max_n=10):
        n = random.randint(0, max_n)
        reps = n * n  # n^2
        return 'AB' * reps

    def test_ab_nen(self, query):
        if len(query) % 2 != 0:
            return False
        pairs = len(query) // 2
        for i in range(pairs):
            if query[2*i:2*i+2] != 'AB':
                return False
        root = int(pairs**0.5)
        return root * root == pairs

    def run_language(self, mode = "observation", query = "none"):
        if mode == "observation":
            return f"{self.language_func()} is generated by the black box"
        elif mode == "intervention":
            if query == "none":
                return f"no query is provided"
            try:
                test_result = self.test_func(query)
                if test_result:
                    return f"{query} is generated by the black box"
                else:
                    return f"{query} cannot be generated by the black box"
            except:
                return f"{query} cannot be generated by the black box"


if __name__ == "__main__":
    print(f"tests")
    language = Languages(language_name="language_2")
    for i in range(10):
        run_language = language.run_language(mode="observation")
        print(f"run_language: {run_language}")

        
        