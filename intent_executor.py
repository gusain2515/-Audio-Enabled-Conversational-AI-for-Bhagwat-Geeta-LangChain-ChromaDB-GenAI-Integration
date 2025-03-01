# intent_executor.py

def detect_intent(query):
    """
    A simple keyword-based intent detection.
    For instance, if the query contains 'appointment', we assume an appointment-related intent.
    """
    if "appointment" in query.lower():
        return "book_appointment"
    return None

def execute_intent(intent, query):
    """
    Execute a function based on the detected intent.
    For example, simulate booking an appointment.
    """
    if intent == "book_appointment":
        # In a real system, parse the query for details and perform booking.
        return "Your appointment has been successfully booked."
    return None
