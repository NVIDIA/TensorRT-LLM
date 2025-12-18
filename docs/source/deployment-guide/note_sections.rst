..
   Reusable note sections for deployment guides.
   Include specific notes using:

   .. include:: note_sections.rst
      :start-after: .. start-note-<name>
      :end-before: .. end-note-<name>

.. start-note-traffic-patterns

.. note::

   **Traffic Patterns**: The ISL (Input Sequence Length) and OSL (Output Sequence Length)
   values in each configuration represent the **maximum supported values** for that config.
   Requests exceeding these limits may result in errors.

   To handle requests with input sequences **longer than the configured ISL**, add the following
   to your config file:

   .. code-block:: yaml

      enable_chunked_prefill: true

   This enables chunked prefill, which processes long input sequences in chunks rather than
   requiring them to fit within a single prefill operation. Note that enabling chunked prefill
   does **not** guarantee optimal performance—these configs are tuned for the specified ISL/OSL.

.. end-note-traffic-patterns

.. start-note-quick-start-isl-osl

.. note::

   The configs here are specifically optimized for a target ISL/OSL (Input/Output Sequence Length) of 1024/1024. If your traffic pattern is different, refer to the :ref:`Comprehensive Configuration Database` section below which covers a larger set of traffic patterns and performance profiles.

.. end-note-quick-start-isl-osl
