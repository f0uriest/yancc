{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}
   {% for item in methods %}
   {%- if item != '__init__' %}

   .. automethod:: {{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
