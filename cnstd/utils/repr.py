# coding: utf-8
# Copyright (C) 2021, [Breezedeus](https://github.com/breezedeus).
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# Credits: adapted from https://github.com/mindee/doctr

__all__ = ['NestedObject']


def _addindent(s_, num_spaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class NestedObject:
    def extra_repr(self) -> str:
        return ''

    def __repr__(self):
        # We treat the extra repr like the sub-object, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        if hasattr(self, '_children_names'):
            for key in self._children_names:
                child = getattr(self, key)
                if isinstance(child, list) and len(child) > 0:
                    child_str = ",\n".join([repr(subchild) for subchild in child])
                    if len(child) > 1:
                        child_str = _addindent(f"\n{child_str},", 2) + '\n'
                    child_str = f"[{child_str}]"
                else:
                    child_str = repr(child)
                child_str = _addindent(child_str, 2)
                child_lines.append('(' + key + '): ' + child_str)
        lines = extra_lines + child_lines

        main_str = self.__class__.__name__ + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str
